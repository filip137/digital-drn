from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

from digital_drn.training.experiment import build_model_from_config  # noqa: E402
from digital_drn.training.loaded_model_beta_sweep import (  # noqa: E402
    _aggregate_sweep,
    _load_checkpoint_into_model,
    _load_json,
    _subset_eval_loader,
)
from digital_drn.training.trainer import build_criterion  # noqa: E402
from digital_drn.utils.misc import resolve_device, set_seed  # noqa: E402


def _empty_stats() -> dict[str, float]:
    return {"sum_sq": 0.0, "sum_abs": 0.0, "numel": 0.0, "max_abs": 0.0}


def _update_stats(stats: dict[str, float], tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float()
    abs_tensor = tensor.abs()
    stats["sum_sq"] += float(torch.sum(tensor * tensor).item())
    stats["sum_abs"] += float(torch.sum(abs_tensor).item())
    stats["numel"] += int(tensor.numel())
    stats["max_abs"] = max(stats["max_abs"], float(abs_tensor.max().item()))


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    numel = float(stats["numel"])
    if numel <= 0:
        return {"rms": float("nan"), "mean_abs": float("nan"), "max_abs": float("nan"), "numel": 0}
    return {
        "rms": math.sqrt(stats["sum_sq"] / numel),
        "mean_abs": stats["sum_abs"] / numel,
        "max_abs": stats["max_abs"],
        "numel": int(numel),
    }


def _collect_residual_currents(model, loader: DataLoader, device: torch.device, *, num_iterations: int) -> dict:
    layer_stats: OrderedDict[str, dict[str, float]] = OrderedDict()
    block_stats: OrderedDict[str, dict[str, float]] = OrderedDict()
    overall = _empty_stats()

    model.eval()
    for batch_inputs, _batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        _ = model(batch_inputs, reset=True, num_iterations=num_iterations)
        for block_idx, block in enumerate(model.blocks):
            block_key = f"block_{block_idx}"
            block_acc = block_stats.setdefault(block_key, _empty_stats())
            for layer_idx, layer in enumerate(block.free_layers()):
                layer_key = f"{block_key}/layer_{layer_idx}"
                layer_acc = layer_stats.setdefault(layer_key, _empty_stats())
                residual = block.energy.grad_layer_fn(layer)()
                _update_stats(layer_acc, residual)
                _update_stats(block_acc, residual)
                _update_stats(overall, residual)
        model.detach_state_()

    finalized_layers = OrderedDict((name, _finalize_stats(stats)) for name, stats in layer_stats.items())
    finalized_blocks = OrderedDict((name, _finalize_stats(stats)) for name, stats in block_stats.items())
    finalized_overall = _finalize_stats(overall)
    worst_by_max = max(finalized_layers.items(), key=lambda item: item[1]["max_abs"])
    worst_by_rms = max(finalized_layers.items(), key=lambda item: item[1]["rms"])
    return {
        "overall": finalized_overall,
        "blocks": finalized_blocks,
        "layers": finalized_layers,
        "worst_layer_by_max_abs": {"layer": worst_by_max[0], **worst_by_max[1]},
        "worst_layer_by_rms": {"layer": worst_by_rms[0], **worst_by_rms[1]},
    }


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = _load_json(config_path)
    model = build_model_from_config(config)
    model.set_device(device)
    _load_checkpoint_into_model(model, checkpoint_path)
    model.enable_resistive_grad_()
    model.eval()
    model.detach_state_()
    return config, model


def _build_residual_loader(config: dict, *, sample_limit: int, batch_size: int) -> tuple[DataLoader, int]:
    eval_loader, sample_count = _subset_eval_loader(config, sample_limit=sample_limit, batch_size=sample_limit)
    loader = DataLoader(
        eval_loader.dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return loader, sample_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep num_iterations on frozen hard-sigmoid checkpoints.")
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=Path("/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_iteration_sweep"),
    )
    parser.add_argument("--iterations", default="4,8,16,32")
    parser.add_argument("--diagnostic-beta", type=float, default=1.0e-2)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--residual-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    set_seed(args.seed, deterministic=False)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_summary = json.loads(args.input_summary.read_text())
    rows = list(input_summary.get("rows", []))
    if not rows:
        raise ValueError(f"No rows found in {args.input_summary}.")

    device = resolve_device(args.device)
    iteration_values = [int(item.strip()) for item in args.iterations.split(",") if item.strip()]

    results = []
    for row in rows:
        checkpoint_dir = Path(row["checkpoint_dir"])
        checkpoint_path = Path(row.get("checkpoint_best") or (checkpoint_dir / "checkpoint_best.pt"))
        config_path = checkpoint_dir / "experiment_config.json"

        config, _ = _load_model(config_path, checkpoint_path, device)
        criterion = build_criterion(config.get("trainer", {}).get("criterion", "cross_entropy")).to(device)
        metric_loader, sample_count = _subset_eval_loader(
            config,
            sample_limit=args.sample_limit,
            batch_size=args.metric_batch_size,
        )
        residual_loader, _ = _build_residual_loader(
            config,
            sample_limit=args.sample_limit,
            batch_size=args.residual_batch_size,
        )
        amp_comp = bool(config.get("algorithm", {}).get("config", {}).get("ad_hoc_amp_gradient_scale", False))

        for num_iterations in iteration_values:
            _config, model = _load_model(config_path, checkpoint_path, device)
            metric_row = _aggregate_sweep(
                model,
                metric_loader,
                criterion,
                [float(args.diagnostic_beta)],
                num_iterations=num_iterations,
                amp_gradient_compensation=amp_comp,
            )[0]
            model.detach_state_()

            _config, residual_model = _load_model(config_path, checkpoint_path, device)
            residuals = _collect_residual_currents(
                residual_model,
                residual_loader,
                device,
                num_iterations=num_iterations,
            )
            results.append(
                {
                    "checkpoint_source": row["algorithm"],
                    "checkpoint_dir": str(checkpoint_dir),
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "num_iterations": int(num_iterations),
                    "sample_count": int(sample_count),
                    "metric_batch_size": int(args.metric_batch_size),
                    "residual_batch_size": int(args.residual_batch_size),
                    "diagnostic_beta": float(args.diagnostic_beta),
                    "overall_cosine": metric_row["overall_cosine"],
                    "overall_relative_error": metric_row["overall_relative_error"],
                    "bp_norm": metric_row["bp_norm"],
                    "ep_norm": metric_row["ep_norm"],
                    "displacement": metric_row["displacement"],
                    "groups": metric_row["groups"],
                    "residual_currents": residuals,
                }
            )
            print(
                "[done]"
                f" checkpoint={row['algorithm']}"
                f" iters={num_iterations}"
                f" cosine={metric_row['overall_cosine']:.4f}"
                f" relerr={metric_row['overall_relative_error']:.4f}"
                f" residual_rms={residuals['overall']['rms']:.6g}"
                f" residual_max={residuals['overall']['max_abs']:.6g}"
            )

    summary = {
        "mode": "frozen_checkpoint_iteration_sweep",
        "input_summary": str(args.input_summary),
        "device": str(device),
        "iterations": iteration_values,
        "diagnostic_beta": float(args.diagnostic_beta),
        "sample_limit": int(args.sample_limit),
        "rows": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
