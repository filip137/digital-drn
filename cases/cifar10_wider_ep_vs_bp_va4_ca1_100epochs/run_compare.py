from __future__ import annotations

import csv
import gc
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch

from digital_drn.training.ep_network import hybrid_backward_explicit
from digital_drn.training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    load_experiment_config,
)
from digital_drn.training.loaded_model_beta_sweep import _aggregate_sweep
from digital_drn.training.trainer import build_criterion


CASE_DIR = Path(__file__).resolve().parent
SUMMARY_JSON_PATH = CASE_DIR / "summary.json"
SUMMARY_CSV_PATH = CASE_DIR / "summary.csv"
GROUP_SUMMARY_JSON_PATH = CASE_DIR / "group_summary.json"
TENSOR_SUMMARY_JSON_PATH = CASE_DIR / "tensor_norm_summary.json"
TENSOR_SUMMARY_CSV_PATH = CASE_DIR / "tensor_norm_summary.csv"
OUTPUT_ROOT = CASE_DIR / "runs"

ALGORITHMS = ("ep", "bp")
DIAGNOSTIC_BETA = 1.0e-2
VOLTAGE_AMP = 4.0
CURRENT_AMP = 1.0


def _configure_algorithm(algorithm: str) -> dict:
    config = load_experiment_config(
        config_name="cifar10_drn_only_signed_norm_readout_wider",
        algorithm=algorithm,
    )

    trainer_cfg = config["trainer"]
    trainer_cfg["epochs"] = 100
    trainer_cfg["log_every"] = 10
    trainer_cfg["save_best"] = False
    trainer_cfg["save_last"] = False
    trainer_cfg["save_events"] = False

    data_cfg = config["data"]["config"]
    data_cfg["batch_size"] = 16
    data_cfg["num_workers"] = 0
    data_cfg["train_subset"] = 128
    data_cfg["test_subset"] = 128

    run_cfg = config["config"]
    run_cfg["device"] = "cuda"
    run_cfg["save"] = True
    run_cfg["output_dir"] = str(OUTPUT_ROOT)

    model_cfg = config["model"]["config"]
    model_cfg["voltage_amp"] = VOLTAGE_AMP
    model_cfg["current_amp"] = CURRENT_AMP

    config["_selection"]["model"] = f"cifar10_wider_va4_ca1_{algorithm}"
    config["model"]["name"] = "cifar10_drn_only_signed_norm_readout_wider_va4_ca1"
    return config


def _history_metrics(history) -> dict[str, float | int | None]:
    train_accuracy = list(history.train_accuracy)
    val_accuracy = list(history.val_accuracy)
    train_loss = list(history.train_loss)
    val_loss = list(history.val_loss)

    best_train = max(train_accuracy) if train_accuracy else None
    best_val = max(val_accuracy) if val_accuracy else None
    return {
        "epochs_recorded": len(train_accuracy),
        "train_final": train_accuracy[-1] if train_accuracy else None,
        "train_best": best_train,
        "train_best_epoch": (train_accuracy.index(best_train) + 1) if train_accuracy else None,
        "val_final": val_accuracy[-1] if val_accuracy else None,
        "val_best": best_val,
        "val_best_epoch": (val_accuracy.index(best_val) + 1) if val_accuracy else None,
        "train_loss_final": train_loss[-1] if train_loss else None,
        "val_loss_final": val_loss[-1] if val_loss else None,
    }


def _write_summary(rows: list[dict]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_bp_named(model) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    if getattr(model, "head", None) is not None:
        for name, param in model.head.named_parameters():
            if not param.requires_grad:
                continue
            grad = param.grad.detach().cpu().float().clone() if param.grad is not None else torch.zeros_like(
                param.detach().cpu().float()
            )
            out[f"head/{name}"] = grad
    for block_idx, block in enumerate(model.blocks):
        for name, param in block.ff.named_parameters():
            if not param.requires_grad:
                continue
            grad = param.grad.detach().cpu().float().clone() if param.grad is not None else torch.zeros_like(
                param.detach().cpu().float()
            )
            out[f"block_{block_idx}/ff/{name}"] = grad
        if block._drive_scale_raw.requires_grad:
            grad = (
                block._drive_scale_raw.grad.detach().cpu().float().clone()
                if block._drive_scale_raw.grad is not None
                else torch.zeros_like(block._drive_scale_raw.detach().cpu().float())
            )
            out[f"block_{block_idx}/drive/drive_scale_raw"] = grad
        for name, state in block.named_resistive_parameters():
            grad = state.grad.detach().cpu().float().clone() if state.grad is not None else torch.zeros_like(
                state.detach().cpu().float()
            )
            out[f"block_{block_idx}/drn/{name}"] = grad
    return out


def _collect_ep_named(model, hybrid) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    if getattr(model, "head", None) is not None:
        head_names = [name for name, param in model.head.named_parameters() if param.requires_grad]
        for name, grad in zip(head_names, hybrid.head.head_param_grads):
            out[f"head/{name}"] = grad.detach().cpu().float().clone()
    for block_idx, (block, block_result) in enumerate(zip(model.blocks, hybrid.blocks)):
        ff_names = [name for name, param in block.ff.named_parameters() if param.requires_grad]
        for name, grad in zip(ff_names, block_result.digital.ff_param_grads):
            out[f"block_{block_idx}/ff/{name}"] = grad.detach().cpu().float().clone()
        if block._drive_scale_raw.requires_grad:
            out[f"block_{block_idx}/drive/drive_scale_raw"] = (
                block_result.digital.drive_scale_grad.detach().cpu().float().clone()
            )
        resistive_names = [name for name, _state in block.named_resistive_parameters()]
        for name, grad in zip(resistive_names, block_result.ep.param_grads):
            out[f"block_{block_idx}/drn/{name}"] = grad.detach().cpu().float().clone()
    return out


def _accumulate_named(
    totals: OrderedDict[str, torch.Tensor] | None,
    batch: OrderedDict[str, torch.Tensor],
    *,
    weight: float,
) -> OrderedDict[str, torch.Tensor]:
    if totals is None:
        return OrderedDict((name, tensor.clone() * weight) for name, tensor in batch.items())
    for name, tensor in batch.items():
        totals[name].add_(tensor, alpha=weight)
    return totals


def _tensor_norm_rows(model, trainer, eval_loader, criterion) -> list[dict]:
    bp_totals = None
    ep_totals = None
    total_samples = 0

    for batch_inputs, batch_targets in eval_loader:
        batch_inputs = batch_inputs.to(trainer.device)
        batch_targets = batch_targets.to(trainer.device)
        batch_size = int(batch_targets.size(0))

        model.zero_grad(set_to_none=True)
        model.zero_resistive_grad_(set_to_none=True)
        logits = model(batch_inputs, reset=True, num_iterations=trainer.config.eval_num_iterations)
        loss = criterion(logits, batch_targets)
        loss.backward()
        bp_named = _collect_bp_named(model)
        model.detach_state_()

        model.zero_grad(set_to_none=True)
        model.zero_resistive_grad_(set_to_none=True)
        hybrid = hybrid_backward_explicit(
            model,
            batch_inputs,
            batch_targets,
            criterion=criterion,
            beta=DIAGNOSTIC_BETA,
            reset=True,
            num_iterations=trainer.config.eval_num_iterations,
        )
        ep_named = _collect_ep_named(model, hybrid)

        bp_totals = _accumulate_named(bp_totals, bp_named, weight=float(batch_size))
        ep_totals = _accumulate_named(ep_totals, ep_named, weight=float(batch_size))
        total_samples += batch_size

        model.zero_grad(set_to_none=True)
        model.zero_resistive_grad_(set_to_none=True)
        model.detach_state_()

    assert bp_totals is not None and ep_totals is not None
    rows = []
    for name in bp_totals.keys():
        bp_mean = bp_totals[name] / float(total_samples)
        ep_mean = ep_totals[name] / float(total_samples)
        bp_norm = float(bp_mean.norm().item())
        ep_norm = float(ep_mean.norm().item())
        rows.append(
            {
                "tensor": name,
                "shape": list(bp_mean.shape),
                "bp_norm": bp_norm,
                "ep_norm": ep_norm,
                "ep_over_bp": ep_norm / bp_norm if bp_norm > 1.0e-12 else float("nan"),
                "relative_gap": abs(bp_norm - ep_norm) / bp_norm if bp_norm > 1.0e-12 else float("nan"),
            }
        )
    rows.sort(key=lambda row: row["relative_gap"], reverse=True)
    return rows


def _write_tensor_summary(results: list[dict]) -> None:
    TENSOR_SUMMARY_JSON_PATH.write_text(json.dumps(results, indent=2))
    with TENSOR_SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["algorithm", "tensor", "shape", "bp_norm", "ep_norm", "ep_over_bp", "relative_gap"],
        )
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                payload = dict(row)
                payload["algorithm"] = result["algorithm"]
                payload["shape"] = json.dumps(payload["shape"])
                writer.writerow(payload)


def main() -> int:
    summary_rows: list[dict] = []
    group_rows: list[dict] = []
    tensor_rows: list[dict] = []

    for algorithm in ALGORITHMS:
        print(f"[start] algorithm={algorithm}")
        started_at = time.time()
        model = trainer = train_loader = eval_loader = history = criterion = None
        try:
            config = _configure_algorithm(algorithm)
            model = build_model_from_config(config)
            trainer = build_trainer_from_config(model, config)
            train_loader, eval_loader = build_dataloaders_from_config(config, download=False)
            history = trainer.fit(train_loader, eval_loader)

            criterion = build_criterion(trainer.config.criterion).to(trainer.device)
            sweep_rows = _aggregate_sweep(
                model,
                eval_loader,
                criterion,
                [DIAGNOSTIC_BETA],
                num_iterations=trainer.config.eval_num_iterations,
            )
            diagnostic = sweep_rows[0]
            displacement = diagnostic["displacement"]
            tensor_norm_rows = _tensor_norm_rows(model, trainer, eval_loader, criterion)

            summary_row = {
                "algorithm": algorithm,
                "voltage_amp": VOLTAGE_AMP,
                "current_amp": CURRENT_AMP,
                "checkpoint_dir": str(trainer.config.checkpoint_dir),
                "seconds": round(time.time() - started_at, 2),
                **_history_metrics(history),
                "diagnostic_beta": DIAGNOSTIC_BETA,
                "displacement_mean": displacement["mean_relative_disp"],
                "displacement_block0_mean": displacement["blocks"][0]["mean_relative_disp"],
                "displacement_block1_mean": displacement["blocks"][1]["mean_relative_disp"],
                "overall_cosine": diagnostic["overall_cosine"],
                "overall_relative_error": diagnostic["overall_relative_error"],
            }
            summary_rows.append(summary_row)

            group_rows.append(
                {
                    "algorithm": algorithm,
                    "voltage_amp": VOLTAGE_AMP,
                    "current_amp": CURRENT_AMP,
                    "diagnostic_beta": DIAGNOSTIC_BETA,
                    "train_best": summary_row["train_best"],
                    "val_best": summary_row["val_best"],
                    "overall_cosine": diagnostic["overall_cosine"],
                    "overall_relative_error": diagnostic["overall_relative_error"],
                    "groups": diagnostic["groups"],
                    "displacement": diagnostic["displacement"],
                }
            )

            tensor_rows.append(
                {
                    "algorithm": algorithm,
                    "voltage_amp": VOLTAGE_AMP,
                    "current_amp": CURRENT_AMP,
                    "train_best": summary_row["train_best"],
                    "val_best": summary_row["val_best"],
                    "rows": tensor_norm_rows,
                }
            )

            _write_summary(summary_rows)
            GROUP_SUMMARY_JSON_PATH.write_text(json.dumps(group_rows, indent=2))
            _write_tensor_summary(tensor_rows)
            print(
                f"[done] algorithm={algorithm} "
                f"train_best={summary_row['train_best']:.4f} "
                f"val_best={summary_row['val_best']:.4f} "
                f"overall_cos={summary_row['overall_cosine']:.4f} "
                f"disp={summary_row['displacement_mean']:.4e}"
            )
        finally:
            if trainer is not None:
                trainer.close()
            del criterion, history, train_loader, eval_loader, trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
