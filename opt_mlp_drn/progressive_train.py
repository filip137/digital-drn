from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoints import (
    load_single_block_checkpoint_dir,
    set_only_active_replaced_layer_trainable,
    trainable_tensors,
)
from .data import load_text_datasets
from .metrics import append_jsonl, drn_saturation_fraction, grad_global_norm, hidden_drift_metrics, save_json, write_csv
from .model import OPTMLPDRNForCausalLM, load_opt_causal_lm, parse_layer_indices
from .teacher import apply_next_layer_norm


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = "char" if args.debug and args.tokenizer == "auto" else args.tokenizer
    train_dataset, val_dataset, _encode, _decode = load_text_datasets(
        args.data,
        model_name=args.model_name,
        tokenizer=tokenizer,
        train_frac=args.train_frac,
        block_size=args.block_size,
        vocab_size=128 if args.debug else None,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    teacher = _build_teacher(args).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    num_layers = int(teacher.config.num_hidden_layers)
    layer_indices = parse_layer_indices(args.layers, num_layers)
    if args.max_layer is not None:
        layer_indices = [layer for layer in layer_indices if layer <= args.max_layer]

    output_dir = Path(args.output_dir) / f"opt_mlp_drn_progressive_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "model_name": args.model_name,
        "experiment": "progressive_replacement",
        "layers": layer_indices,
        "steps_per_layer": args.steps_per_layer,
        "next_ln_alpha": args.next_ln_alpha,
        "drn_signed_drive": args.drn_signed_drive,
        "drn_drive_architecture": args.drn_drive_architecture,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "output_dir": str(output_dir),
        "checkpoint_paths": {},
    }
    save_json(output_dir / "run_metadata.json", metadata)

    rows = []
    progressive_block_dir = output_dir / "progressive_single_blocks"
    for layer_index in layer_indices:
        row = _train_layer(
            args,
            teacher,
            train_loader,
            val_loader,
            layer_index,
            output_dir,
            progressive_block_dir,
            device,
        )
        rows.append(row)
        metadata["checkpoint_paths"][str(layer_index)] = row["checkpoint_path"]
        metadata.setdefault("single_block_checkpoint_paths", {})[str(layer_index)] = row["single_block_checkpoint_path"]
        save_json(output_dir / "run_metadata.json", metadata)

    write_csv(output_dir / "progressive_summary.csv", rows)
    save_json(output_dir / "progressive_summary.json", rows)
    save_json(output_dir / "final_metrics.json", {"layers": rows, "output_dir": str(output_dir)})
    print(json.dumps({"layers": rows, "output_dir": str(output_dir)}, indent=2, sort_keys=True))


def _train_layer(
    args,
    teacher,
    train_loader,
    val_loader,
    layer_index: int,
    output_dir: Path,
    progressive_block_dir: Path,
    device: torch.device,
):
    layer_dir = output_dir / f"layer_{layer_index}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    model = _build_student(args, teacher, replace_layers=list(range(layer_index + 1))).to(device)
    if args.checkpoint_dir is not None:
        load_single_block_checkpoint_dir(model, args.checkpoint_dir, layers=list(range(layer_index + 1)))
    if layer_index > 0 and progressive_block_dir.exists():
        load_single_block_checkpoint_dir(model, progressive_block_dir, layers=list(range(layer_index)))
    set_only_active_replaced_layer_trainable(model, layer_index)
    params = trainable_tensors(model)
    if not params:
        raise RuntimeError(f"Layer {layer_index} has no trainable DRN tensors.")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    train_iter = _cycle(train_loader)
    metrics_path = layer_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    initial = _evaluate(model, teacher, val_loader, layer_index, device, args)
    best_loss = initial["loss"]
    append_jsonl(metrics_path, {"stage": "eval", "step": 0, "layer": layer_index, **initial})
    last_grad_norm = 0.0
    start_time = time.time()
    for step in range(1, args.steps_per_layer + 1):
        model.train()
        x, _y = next(train_iter)
        x = x.to(device)
        loss, _metrics = _progressive_loss(model, teacher, x, layer_index, args.next_ln_alpha)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = grad_global_norm(params)
        if args.grad_clip > 0.0:
            clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        model.clamp_resistive_params_()
        model.detach_state_()
        model.clear_distillation_caches()
        append_jsonl(metrics_path, {"stage": "train", "step": step, "layer": layer_index, "loss": float(loss.detach().item()), "grad_norm": last_grad_norm})

        if step % args.eval_interval == 0 or step == args.steps_per_layer:
            metrics = _evaluate(model, teacher, val_loader, layer_index, device, args)
            best_loss = min(best_loss, metrics["loss"])
            append_jsonl(metrics_path, {"stage": "eval", "step": step, "layer": layer_index, **metrics})
            print({"stage": "eval", "step": step, "layer": layer_index, **metrics})

    final = _evaluate(model, teacher, val_loader, layer_index, device, args)
    final.update(
        {
            "layer": layer_index,
            "initial_loss": initial["loss"],
            "best_loss": best_loss,
            "last_train_grad_norm": float(last_grad_norm),
            "training_time_sec": time.time() - start_time,
            "trainable_params": sum(t.numel() for t in params),
            "peak_memory_mb": _cuda_peak_memory_mb(),
        }
    )
    checkpoint_path = layer_dir / "checkpoint_last.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "metrics": final}, checkpoint_path)
    final["checkpoint_path"] = str(checkpoint_path)
    single_block_checkpoint_path = progressive_block_dir / f"layer_{layer_index}" / "checkpoint_last.pt"
    _save_active_single_block_checkpoint(single_block_checkpoint_path, model, layer_index, optimizer, args, final)
    final["single_block_checkpoint_path"] = str(single_block_checkpoint_path)
    save_json(layer_dir / "final_metrics.json", final)
    return final


def _save_active_single_block_checkpoint(
    path: Path,
    model: OPTMLPDRNForCausalLM,
    layer_index: int,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for layer in model.replaced_layers():
        if layer.layer_index != layer_index:
            continue
        state = {f"drn.{key}": value.detach().cpu() for key, value in layer.drn_mlp.state_dict().items()}
        state["input_scale"] = layer.drn_input_scale.detach().cpu()
        state["output_scale"] = layer.drn_output_scale.detach().cpu()
        state["output_gain"] = layer.drn_output_gain.detach().cpu()
        torch.save(
            {
                "model": state,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "metrics": metrics,
                "layer_index": int(layer_index),
            },
            path,
        )
        return
    raise RuntimeError(f"Could not find active replaced layer {layer_index} to checkpoint.")


@torch.no_grad()
def _evaluate(model, teacher, loader, layer_index: int, device: torch.device, args) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows = []
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= args.eval_iters:
            break
        loss, metrics = _progressive_loss(model, teacher, x.to(device), layer_index, args.next_ln_alpha)
        rows.append({"loss": float(loss.detach().item()), **metrics})
        model.detach_state_()
        model.clear_distillation_caches()
    if was_training:
        model.train()
    return _mean_rows(rows)


def _progressive_loss(model, teacher, input_ids: torch.Tensor, layer_index: int, next_ln_alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        teacher_out = teacher(input_ids=input_ids, labels=None, use_cache=False, output_hidden_states=True, return_dict=True)
        teacher_hidden = list(teacher_out.hidden_states)
    student_out = model(input_ids, return_hidden_states=True)
    student_hidden = student_out["hidden_states"]
    if student_hidden is None:
        raise RuntimeError("Student did not return hidden states.")
    target_depth = layer_index + 1
    hidden_loss = F.mse_loss(student_hidden[target_depth], teacher_hidden[target_depth].detach())
    loss = hidden_loss
    metrics = {
        "hidden_mse": float(hidden_loss.detach().item()),
        "saturation_fraction": _layer_saturation(model, layer_index),
    }
    if next_ln_alpha > 0.0:
        teacher_next = apply_next_layer_norm(teacher, layer_index, teacher_hidden[target_depth].detach())
        student_next = apply_next_layer_norm(teacher, layer_index, student_hidden[target_depth])
        next_ln_loss = F.mse_loss(student_next, teacher_next)
        loss = loss + float(next_ln_alpha) * next_ln_loss
        metrics["next_ln_mse"] = float(next_ln_loss.detach().item())
    metrics.update(hidden_drift_metrics(student_hidden, teacher_hidden, max_depth=target_depth, prefix="hidden"))
    return loss, metrics


def _layer_saturation(model, layer_index: int) -> float:
    for layer in model.replaced_layers():
        if layer.layer_index == layer_index:
            return drn_saturation_fraction(layer)
    return float("nan")


def _build_student(args, teacher, replace_layers: list[int]) -> OPTMLPDRNForCausalLM:
    return OPTMLPDRNForCausalLM(
        copy.deepcopy(teacher).cpu(),
        replace_mlp_layers=replace_layers,
        drn_iter=args.drn_iter,
        signed_drive=args.drn_signed_drive,
        drive_architecture=args.drn_drive_architecture,
        non_linearity=args.drn_non_linearity,
        hidden_multiplier=args.drn_hidden_multiplier,
        weight_gains=args.drn_weight_gains,
        weight_min=args.drn_weight_min,
        weight_max=args.drn_weight_max,
        hard_sigmoid_param=_hard_sigmoid_param(args),
        bias_gain=args.drn_bias_gain,
        init_drive_scale=args.drn_init_drive_scale,
        voltage_amp=args.drn_voltage_amp,
        current_amp=args.drn_current_amp,
        learn_amplification=args.drn_learn_amplification,
    )


def _build_teacher(args) -> torch.nn.Module:
    if not args.debug:
        return load_opt_causal_lm(args.model_name)
    from transformers import OPTConfig, OPTForCausalLM

    cfg = OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=3,
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
    return OPTForCausalLM(cfg)


def _hard_sigmoid_param(args) -> dict[str, float] | None:
    if args.drn_non_linearity != "hard_sigmoid":
        return None
    return {"g_on": args.drn_hard_sigmoid_g_on, "g_off": args.drn_hard_sigmoid_g_off, "v_min": args.drn_hard_sigmoid_v_min, "v_max": args.drn_hard_sigmoid_v_max}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max_layer", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--steps_per_layer", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--next_ln_alpha", type=float, default=0.0)
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
    parser.add_argument("--drn_non_linearity", choices=["perfect_diode", "hard_sigmoid", "linear", "lpw_diode"], default="perfect_diode")
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
    if args.steps_per_layer <= 0 or args.eval_interval <= 0 or args.eval_iters <= 0:
        raise ValueError("step and eval counts must be positive.")
    return args


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int))})
    return {key: float(sum(float(row[key]) for row in rows if key in row) / sum(1 for row in rows if key in row)) for key in keys}


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(raw: str | None) -> torch.device:
    return torch.device(raw) if raw else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_peak_memory_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() else 0.0


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
