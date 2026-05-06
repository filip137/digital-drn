from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import load_text_datasets
from .model import OPTMLPDRNForCausalLM


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    train_dataset, val_dataset, _encode, _decode = load_text_datasets(
        args.data,
        model_name=args.model_name,
        tokenizer="char" if args.debug and args.tokenizer == "auto" else args.tokenizer,
        train_frac=args.train_frac,
        block_size=args.block_size,
        vocab_size=128 if args.debug else None,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    train_iter = _cycle(train_loader)

    model = _build_model(args).to(device)
    model.enable_resistive_grad_(True)
    trainable_tensors = _trainable_tensors(model)
    if not trainable_tensors:
        raise RuntimeError("No trainable DRN tensors found.")

    output_dir = Path(args.output_dir) / f"opt_mlp_drn_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[str, str] = {}
    metadata = {
        "model_name": args.model_name,
        "objective": args.objective,
        "replace_mlp_layers": args.replace_mlp_layers,
        "replaced_layer_indices": model.replaced_layer_indices,
        "distill_steps": args.distill_steps,
        "ce_steps": args.ce_steps,
        "drn_iter": args.drn_iter,
        "drn_signed_drive": args.drn_signed_drive,
        "drn_drive_architecture": args.drn_drive_architecture,
        "drn_hidden_multiplier": args.drn_hidden_multiplier,
        "trainable_params": _count_tensors(trainable_tensors),
        "total_params": _count_tensors(_all_model_tensors(model)),
        "output_dir": str(output_dir),
        "checkpoint_paths": checkpoint_paths,
    }
    _save_json(output_dir / "run_metadata.json", metadata)

    final_metrics: dict[str, Any] = {}
    if args.objective in {"residual_distill", "distill_then_ce"}:
        distill_metrics = _run_distillation_stage(
            model=model,
            train_iter=train_iter,
            val_loader=val_loader,
            device=device,
            args=args,
            output_dir=output_dir,
        )
        final_metrics.update({f"distill/{key}": value for key, value in distill_metrics.items()})
        checkpoint_paths["distilled"] = str(output_dir / "checkpoint_distilled.pt")
        _save_json(output_dir / "run_metadata.json", metadata)

    if args.objective in {"ce", "distill_then_ce"}:
        ce_metrics = _run_ce_stage(
            model=model,
            train_iter=train_iter,
            val_loader=val_loader,
            device=device,
            args=args,
            output_dir=output_dir,
        )
        final_metrics.update({f"ce/{key}": value for key, value in ce_metrics.items()})
        checkpoint_paths["last"] = str(output_dir / "checkpoint_last.pt")
        _save_json(output_dir / "run_metadata.json", metadata)

    final_metrics["peak_memory_mb"] = _cuda_peak_memory_mb()
    final_metrics["checkpoint_paths"] = checkpoint_paths
    _save_json(output_dir / "final_metrics.json", final_metrics)
    print(final_metrics)


def _run_distillation_stage(
    *,
    model: OPTMLPDRNForCausalLM,
    train_iter,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    optimizer = _make_optimizer(model, args)
    initial_eval = _evaluate_distillation(model, val_loader, device, max_batches=args.eval_iters)
    best_loss = initial_eval["distill_loss"]
    print({"stage": "distill", "step": 0, **initial_eval, "peak_memory_mb": _cuda_peak_memory_mb()})

    for step in range(1, args.distill_steps + 1):
        model.train()
        x, _y = next(train_iter)
        x = x.to(device)
        result = model.distillation_loss(x)
        loss = result["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            clip_grad_norm_(_trainable_tensors(model), args.grad_clip)
        optimizer.step()
        model.clamp_resistive_params_()
        model.detach_state_()
        model.clear_distillation_caches()

        if step % args.eval_interval == 0 or step == args.distill_steps:
            metrics = _evaluate_distillation(model, val_loader, device, max_batches=args.eval_iters)
            best_loss = min(best_loss, metrics["distill_loss"])
            print({"stage": "distill", "step": step, **metrics, "peak_memory_mb": _cuda_peak_memory_mb()})

    metrics = _evaluate_distillation(model, val_loader, device, max_batches=args.eval_iters)
    metrics["initial_distill_loss"] = initial_eval["distill_loss"]
    metrics["best_distill_loss"] = best_loss
    _save_checkpoint(output_dir / "checkpoint_distilled.pt", model, optimizer, args.distill_steps, args, metrics)
    _save_json(output_dir / "distill_metrics.json", metrics)
    return metrics


def _run_ce_stage(
    *,
    model: OPTMLPDRNForCausalLM,
    train_iter,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    optimizer = _make_optimizer(model, args)
    initial_eval = _evaluate_ce(model, val_loader, device, max_batches=args.eval_iters)
    best_loss = initial_eval["loss"]
    print({"stage": "ce", "step": 0, **initial_eval, "peak_memory_mb": _cuda_peak_memory_mb()})

    for step in range(1, args.ce_steps + 1):
        model.train()
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)
        out = model(x, targets=y)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("Model did not return a CE loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            clip_grad_norm_(_trainable_tensors(model), args.grad_clip)
        optimizer.step()
        model.clamp_resistive_params_()
        model.detach_state_()
        model.clear_distillation_caches()

        if step % args.eval_interval == 0 or step == args.ce_steps:
            metrics = _evaluate_ce(model, val_loader, device, max_batches=args.eval_iters)
            best_loss = min(best_loss, metrics["loss"])
            print({"stage": "ce", "step": step, **metrics, "peak_memory_mb": _cuda_peak_memory_mb()})

    metrics = _evaluate_ce(model, val_loader, device, max_batches=args.eval_iters)
    metrics["initial_loss"] = initial_eval["loss"]
    metrics["best_loss"] = best_loss
    _save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, args.ce_steps, args, metrics)
    _save_json(output_dir / "ce_metrics.json", metrics)
    return metrics


@torch.no_grad()
def _evaluate_distillation(
    model: OPTMLPDRNForCausalLM,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        result = model.distillation_loss(x.to(device))
        metrics = result["metrics"]
        rows.append({key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))})
        model.clear_distillation_caches()
        model.detach_state_()
    if was_training:
        model.train()
    return _mean_rows(rows)


@torch.no_grad()
def _evaluate_ce(
    model: OPTMLPDRNForCausalLM,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        out = model(x.to(device), targets=y.to(device))
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("Model did not return a CE loss during evaluation.")
        losses.append(float(loss.item()))
        model.clear_distillation_caches()
        model.detach_state_()
    if was_training:
        model.train()
    loss = sum(losses) / max(1, len(losses))
    return {"loss": loss, "ppl": _perplexity(loss)}


def _hard_sigmoid_param(args: argparse.Namespace) -> dict[str, float] | None:
    if args.drn_non_linearity != "hard_sigmoid":
        return None
    return {
        "g_on": float(args.drn_hard_sigmoid_g_on),
        "g_off": float(args.drn_hard_sigmoid_g_off),
        "v_min": float(args.drn_hard_sigmoid_v_min),
        "v_max": float(args.drn_hard_sigmoid_v_max),
    }


def _build_model(args: argparse.Namespace) -> OPTMLPDRNForCausalLM:
    kwargs = {
        "replace_mlp_layers": args.replace_mlp_layers,
        "drn_iter": args.drn_iter,
        "signed_drive": args.drn_signed_drive,
        "drive_architecture": args.drn_drive_architecture,
        "non_linearity": args.drn_non_linearity,
        "hidden_multiplier": args.drn_hidden_multiplier,
        "weight_gains": args.drn_weight_gains,
        "weight_min": args.drn_weight_min,
        "weight_max": args.drn_weight_max,
        "hard_sigmoid_param": _hard_sigmoid_param(args),
        "bias_gain": args.drn_bias_gain,
        "init_drive_scale": args.drn_init_drive_scale,
        "voltage_amp": args.drn_voltage_amp,
        "current_amp": args.drn_current_amp,
        "learn_amplification": args.drn_learn_amplification,
    }
    if args.debug:
        try:
            from transformers import OPTConfig
        except ImportError as exc:
            raise ImportError("Install transformers to construct debug OPT models.") from exc

        cfg = OPTConfig(
            vocab_size=128,
            hidden_size=32,
            ffn_dim=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=args.block_size,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            activation_function="relu",
            do_layer_norm_before=True,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=2,
            word_embed_proj_dim=32,
        )
        return OPTMLPDRNForCausalLM.from_config(cfg, **kwargs)
    return OPTMLPDRNForCausalLM.from_pretrained(args.model_name, **kwargs)


def _make_optimizer(model: OPTMLPDRNForCausalLM, args: argparse.Namespace) -> torch.optim.Optimizer:
    params = _trainable_tensors(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--objective", choices=["ce", "residual_distill", "distill_then_ce"], default="distill_then_ce")
    parser.add_argument("--replace_mlp_layers", default="last:1")
    parser.add_argument("--distill_steps", type=int, default=2000)
    parser.add_argument("--ce_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--drn_signed_drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drn_drive_architecture",
        choices=["projected_hidden", "signed_input_free"],
        default="projected_hidden",
    )
    parser.add_argument(
        "--drn_non_linearity",
        choices=["perfect_diode", "hard_sigmoid", "linear", "lpw_diode"],
        default="perfect_diode",
    )
    parser.add_argument("--drn_hard_sigmoid_g_on", type=float, default=10.0)
    parser.add_argument("--drn_hard_sigmoid_g_off", type=float, default=1.0e-7)
    parser.add_argument("--drn_hard_sigmoid_v_min", type=float, default=-1.2)
    parser.add_argument("--drn_hard_sigmoid_v_max", type=float, default=1.2)
    parser.add_argument("--drn_hidden_multiplier", type=float, default=None)
    parser.add_argument("--drn_weight_gains", type=float, default=0.1)
    parser.add_argument("--drn_weight_min", type=float, default=1.0e-5)
    parser.add_argument("--drn_weight_max", type=float, default=None)
    parser.add_argument("--drn_bias_gain", type=float, default=0.0)
    parser.add_argument("--drn_init_drive_scale", type=float, default=1.0)
    parser.add_argument("--drn_voltage_amp", type=float, default=1.0)
    parser.add_argument("--drn_current_amp", type=float, default=1.0)
    parser.add_argument("--drn_learn_amplification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.distill_steps < 0 or args.ce_steps < 0:
        raise ValueError("Step counts must be non-negative.")
    if args.objective == "residual_distill" and args.distill_steps == 0:
        raise ValueError("--distill_steps must be positive for residual_distill.")
    if args.objective == "ce" and args.ce_steps == 0:
        raise ValueError("--ce_steps must be positive for ce.")
    if args.drn_voltage_amp <= 0.0:
        raise ValueError("--drn_voltage_amp must be positive.")
    if args.drn_current_amp <= 0.0:
        raise ValueError("--drn_current_amp must be positive.")
    return args


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _get_device(raw: str | None) -> torch.device:
    if raw is not None:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _perplexity(loss: float) -> float:
    if loss > 50.0:
        return float("inf")
    return float(math.exp(loss))


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row))
        for key in keys
    }


def _all_model_tensors(model: OPTMLPDRNForCausalLM) -> list[torch.Tensor]:
    return _dedupe_tensors(list(model.parameters()) + model.resistive_param_states())


def _trainable_tensors(model: OPTMLPDRNForCausalLM) -> list[torch.Tensor]:
    return [tensor for tensor in _dedupe_tensors(model.optimizer_tensors()) if tensor.requires_grad]


def _dedupe_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    unique = []
    seen = set()
    for tensor in tensors:
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tensor)
    return unique


def _count_tensors(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)


def _save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(
    path: Path,
    model: OPTMLPDRNForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
        "metrics": metrics,
        "resistive_parameters": {
            name: tensor.detach().cpu() for name, tensor in model.named_resistive_parameters()
        },
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
