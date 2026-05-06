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

from .cache_activations import (
    CachedLayerActivationDataset,
    batch_to_activations,
    calibration_from_cache,
    load_cache_metadata,
)
from .checkpoints import load_single_block_checkpoint_into_single_block
from .calibration import calibrate_teacher, load_calibration, save_calibration, scales_from_calibration
from .data import load_text_datasets
from .metrics import grad_global_norm, write_csv
from .model import load_opt_causal_lm
from .model import OPTMLPDRNForCausalLM
from .single_block import (
    all_tensors,
    build_single_block_drn,
    optimizer_param_groups,
    single_block_loss,
    trainable_tensors,
)
from .teacher import TeacherLayerActivations, collect_teacher_layer_activations, default_probe_layers, opt_num_layers


OBJECTIVES = (
    "local_mlp",
    "local_mlp_cosine",
    "drift_compensated_residual_cosine",
    "post_residual",
    "next_ln_aux",
    "rigorous_pretrain",
)


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    if args.activation_cache is not None and args.eval_logit_kl_batches > 0:
        raise ValueError("--eval_logit_kl_batches requires token data; set it to 0 when using --activation_cache.")

    train_loader = None
    val_loader = None
    cal_loader = None
    if args.activation_cache is None:
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
        cal_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    teacher = _build_teacher(args).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    layer_indices = _parse_layers(args.layers, opt_num_layers(teacher))
    if args.activation_cache is not None:
        _validate_activation_cache(args, layer_indices)
    output_dir = Path(args.output_dir) / f"opt_mlp_drn_single_block_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = _prepare_calibration(args, teacher, cal_loader, layer_indices, device, output_dir)
    metadata: dict[str, Any] = {
        "model_name": args.model_name,
        "experiment": "single_block_mlp_drn_distillation",
        "layers": layer_indices,
        "objective": args.objective,
        "objective_schedule": args.objective_schedule,
        "mlp_input_mode": args.mlp_input_mode,
        "activation_cache": str(args.activation_cache) if args.activation_cache is not None else None,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "input_noise_std": args.input_noise_std,
        "residual_drift_std": args.residual_drift_std,
        "input_noise_mode": args.input_noise_mode,
        "noise_train_only": args.noise_train_only,
        "alpha_next_ln": args.alpha_next_ln,
        "alpha_cosine": args.alpha_cosine,
        "alpha_norm": args.alpha_norm,
        "alpha_post_residual": args.alpha_post_residual,
        "alpha_logit_kl": args.alpha_logit_kl,
        "logit_temperature": args.logit_temperature,
        "init_mode": args.init_mode,
        "teacher_init_scope": _teacher_init_scope(args.init_mode),
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "drn_iter": args.drn_iter,
        "drn_signed_drive": args.drn_signed_drive,
        "drn_drive_architecture": args.drn_drive_architecture,
        "drn_signed_output_weights": args.drn_signed_output_weights,
        "drn_non_linearity": args.drn_non_linearity,
        "hard_sigmoid_param": _hard_sigmoid_param(args),
        "drn_learn_drive_scale": args.drn_learn_drive_scale,
        "drn_train_current_frontend": args.drn_train_current_frontend,
        "drn_hidden_multiplier": args.drn_hidden_multiplier,
        "drn_weight_min": args.drn_weight_min,
        "drn_weight_max": args.drn_weight_max,
        "drn_voltage_amp": args.drn_voltage_amp,
        "drn_current_amp": args.drn_current_amp,
        "drn_learn_amplification": args.drn_learn_amplification,
        "drn_amp_lr": args.drn_amp_lr,
        "output_gain_lr": args.output_gain_lr,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "calibration": calibration,
        "checkpoint_paths": {},
        "best_checkpoint_paths": {},
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "run_metadata.json", metadata)

    final_rows = []
    for layer_index in layer_indices:
        row = _train_one_layer(
            args=args,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            calibration=calibration,
            layer_index=layer_index,
            output_dir=output_dir / f"layer_{layer_index}",
            device=device,
        )
        final_rows.append(row)
        metadata["checkpoint_paths"][str(layer_index)] = row["checkpoint_path"]
        metadata["best_checkpoint_paths"][str(layer_index)] = row.get("best_checkpoint_path")
        _save_json(output_dir / "run_metadata.json", metadata)

    summary = {
        "layers": final_rows,
        "peak_memory_mb": _cuda_peak_memory_mb(),
        "output_dir": str(output_dir),
        "layer_summary_path": str(output_dir / "layer_summary.csv"),
    }
    write_csv(output_dir / "layer_summary.csv", final_rows)
    _save_json(output_dir / "layer_summary.json", final_rows)
    _save_json(output_dir / "final_metrics.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _train_one_layer(
    *,
    args: argparse.Namespace,
    teacher: torch.nn.Module,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    calibration: dict[str, Any] | None,
    layer_index: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.activation_cache is not None:
        train_loader = _cached_loader(args, split="train", layer_index=layer_index, shuffle=True)
        val_loader = _cached_loader(args, split="val", layer_index=layer_index, shuffle=False)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Training and validation loaders must be available.")
    input_scale, output_scale = _layer_scales(args, calibration, layer_index)
    teacher_layer = teacher.model.decoder.layers[layer_index]
    model = build_single_block_drn(
        teacher_layer,
        input_scale=input_scale,
        output_scale=output_scale,
        drn_iter=args.drn_iter,
        signed_drive=args.drn_signed_drive,
        drive_architecture=args.drn_drive_architecture,
        non_linearity=args.drn_non_linearity,
        hidden_multiplier=args.drn_hidden_multiplier,
        weight_gains=args.drn_weight_gains,
        weight_min=args.drn_weight_min,
        weight_max=args.drn_weight_max,
        hard_sigmoid_param=_hard_sigmoid_param(args),
        signed_output_weights=args.drn_signed_output_weights,
        bias_gain=args.drn_bias_gain,
        init_drive_scale=args.drn_init_drive_scale,
        voltage_amp=args.drn_voltage_amp,
        current_amp=args.drn_current_amp,
        learn_amplification=args.drn_learn_amplification,
        learn_drive_scale=args.drn_learn_drive_scale,
        init_mode=_resolved_init_mode(args.init_mode),
    ).to(device)
    model.enable_resistive_grad_(True)
    model.train_current_frontend_(args.drn_train_current_frontend)
    if args.init_checkpoint is not None:
        load_single_block_checkpoint_into_single_block(args.init_checkpoint, model)

    params = trainable_tensors(model)
    if not params:
        raise RuntimeError(f"Layer {layer_index} has no trainable DRN tensors.")
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, amp_lr=args.drn_amp_lr, output_gain_lr=args.output_gain_lr),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_iter = _cycle(train_loader)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    initial_metrics = _evaluate(
        model=model,
        teacher=teacher,
        loader=val_loader,
        layer_index=layer_index,
        device=device,
        args=args,
        objective=_objective_for_step(args, 1),
    )
    best_loss = initial_metrics["loss"]
    best_step = 0
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    last_train_grad_norm = 0.0
    _append_jsonl(metrics_path, {"stage": "eval", "step": 0, "layer": layer_index, **initial_metrics})
    print({"layer": layer_index, "stage": "eval", "step": 0, **initial_metrics})
    _save_checkpoint(best_checkpoint_path, model, optimizer, 0, args, initial_metrics)

    start_time = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = next(train_iter)
        activations = _activations_from_batch(
            batch,
            teacher=teacher,
            layer_index=layer_index,
            device=device,
            args=args,
        )
        activations = _augment_activations(activations, args, training=True)
        result = single_block_loss(
            model,
            teacher,
            layer_index,
            activations,
            objective=_objective_for_step(args, step),
            alpha_next_ln=args.alpha_next_ln,
            alpha_cosine=args.alpha_cosine,
            alpha_norm=args.alpha_norm,
            alpha_post_residual=args.alpha_post_residual,
            alpha_logit_kl=args.alpha_logit_kl,
            logit_temperature=args.logit_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        last_train_grad_norm = grad_global_norm(params)
        if args.grad_clip > 0.0:
            clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        model.clamp_resistive_params_()
        model.detach_state_()
        _append_jsonl(
            metrics_path,
            {
                "stage": "train",
                "step": step,
                "layer": layer_index,
                "objective": _objective_for_step(args, step),
                "loss": float(result.loss.detach().item()),
                "grad_norm": float(last_train_grad_norm),
            },
        )

        if step % args.eval_interval == 0 or step == args.steps:
            eval_objective = _objective_for_step(args, step)
            metrics = _evaluate(
                model=model,
                teacher=teacher,
                loader=val_loader,
                layer_index=layer_index,
                device=device,
                args=args,
                objective=eval_objective,
            )
            if math.isfinite(metrics.get("loss", float("nan"))) and metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_step = step
                _save_checkpoint(best_checkpoint_path, model, optimizer, step, args, metrics)
            payload = {"stage": "eval", "step": step, "layer": layer_index, **metrics}
            _append_jsonl(metrics_path, payload)
            print(payload)

    final_metrics = _evaluate(
        model=model,
        teacher=teacher,
        loader=val_loader,
        layer_index=layer_index,
        device=device,
        args=args,
        objective=_objective_for_step(args, args.steps),
    )
    final_metrics.update(
        {
            "initial_loss": initial_metrics["loss"],
            "initial_rel_mse": initial_metrics["rel_mse"],
            "best_loss": best_loss,
            "best_step": best_step,
            "training_time_sec": time.time() - start_time,
            "peak_memory_mb": _cuda_peak_memory_mb(),
            "solver_iterations": int(args.drn_iter),
            "trainable_params": _count_tensors(params),
            "all_drn_tensors": _count_tensors(all_tensors(model)),
            "last_train_grad_norm": float(last_train_grad_norm),
        }
    )
    if args.eval_logit_kl_batches > 0:
        final_metrics.update(
            _evaluate_replacement_kl(
                block_model=model,
                teacher=teacher,
                loader=val_loader,
                layer_index=layer_index,
                device=device,
                args=args,
            )
        )
    checkpoint_path = output_dir / "checkpoint_last.pt"
    _save_checkpoint(checkpoint_path, model, optimizer, args.steps, args, final_metrics)
    final_metrics["checkpoint_path"] = str(checkpoint_path)
    final_metrics["best_checkpoint_path"] = str(best_checkpoint_path)
    _save_json(output_dir / "final_metrics.json", final_metrics)
    return {"layer": layer_index, **final_metrics}


@torch.no_grad()
def _evaluate(
    *,
    model,
    teacher,
    loader: DataLoader,
    layer_index: int,
    device: torch.device,
    args: argparse.Namespace,
    objective: str | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.eval_iters:
            break
        activations = _activations_from_batch(
            batch,
            teacher=teacher,
            layer_index=layer_index,
            device=device,
            args=args,
        )
        activations = _augment_activations(activations, args, training=False)
        result = single_block_loss(
            model,
            teacher,
            layer_index,
            activations,
            objective=args.objective if objective is None else objective,
            alpha_next_ln=args.alpha_next_ln,
            alpha_cosine=args.alpha_cosine,
            alpha_norm=args.alpha_norm,
            alpha_post_residual=args.alpha_post_residual,
            alpha_logit_kl=args.alpha_logit_kl,
            logit_temperature=args.logit_temperature,
        )
        row = {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (float, int))
        }
        rows.append(row)
        model.detach_state_()
    if was_training:
        model.train()
    metrics = _mean_rows(rows)
    metrics["peak_memory_mb"] = _cuda_peak_memory_mb()
    metrics["solver_iterations"] = float(args.drn_iter)
    return metrics


def _activations_from_batch(
    batch: Any,
    *,
    teacher,
    layer_index: int,
    device: torch.device,
    args: argparse.Namespace,
):
    if isinstance(batch, dict):
        return batch_to_activations(batch, device=device)
    x, _y = batch
    return collect_teacher_layer_activations(
        teacher,
        x.to(device),
        [layer_index],
        mlp_input_mode=args.mlp_input_mode,
    )[layer_index]


def _augment_activations(
    activations: TeacherLayerActivations,
    args: argparse.Namespace,
    *,
    training: bool,
) -> TeacherLayerActivations:
    input_noise_std = float(getattr(args, "input_noise_std", 0.0))
    residual_drift_std = float(getattr(args, "residual_drift_std", 0.0))
    if input_noise_std == 0.0 and residual_drift_std == 0.0:
        return activations
    if bool(getattr(args, "noise_train_only", True)) and not training:
        return activations
    mode = str(getattr(args, "input_noise_mode", "gaussian"))
    if mode != "gaussian":
        raise ValueError(f"Unsupported input_noise_mode '{mode}'.")

    z = activations.z
    a = activations.a
    if input_noise_std > 0.0:
        z_scale = _activation_std(activations.z)
        z = z + torch.randn_like(z) * (input_noise_std * z_scale)
    if residual_drift_std > 0.0:
        a_scale = _activation_std(activations.a)
        a = a + torch.randn_like(a) * (residual_drift_std * a_scale)
    return TeacherLayerActivations(
        z=z,
        r=activations.r,
        a=a,
        h_next=activations.h_next,
        next_ln=activations.next_ln,
    )


def _activation_std(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().std(unbiased=False).clamp_min(1.0e-12).to(
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _cached_loader(args: argparse.Namespace, *, split: str, layer_index: int, shuffle: bool) -> DataLoader:
    if args.activation_cache is None:
        raise RuntimeError("No activation cache configured.")
    dataset = CachedLayerActivationDataset(
        args.activation_cache,
        split=split,
        layer_index=layer_index,
        max_open_shards=args.activation_cache_open_shards,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.activation_cache_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate_replacement_kl(
    *,
    block_model,
    teacher,
    loader: DataLoader,
    layer_index: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    student_base = copy.deepcopy(teacher).to(device)
    student = OPTMLPDRNForCausalLM(
        student_base,
        replace_mlp_layers=[layer_index],
        drn_iter=args.drn_iter,
        signed_drive=args.drn_signed_drive,
        drive_architecture=args.drn_drive_architecture,
        non_linearity=args.drn_non_linearity,
        hidden_multiplier=args.drn_hidden_multiplier,
        weight_gains=args.drn_weight_gains,
        weight_min=args.drn_weight_min,
        weight_max=args.drn_weight_max,
        hard_sigmoid_param=_hard_sigmoid_param(args),
        signed_output_weights=args.drn_signed_output_weights,
        bias_gain=args.drn_bias_gain,
        init_drive_scale=args.drn_init_drive_scale,
        voltage_amp=args.drn_voltage_amp,
        current_amp=args.drn_current_amp,
        learn_amplification=args.drn_learn_amplification,
    ).to(device)
    _copy_single_block_into_wrapper(block_model, student.replaced_layers()[0])
    teacher.eval()
    student.eval()

    rows = []
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= args.eval_logit_kl_batches:
            break
        x = x.to(device)
        teacher_logits = teacher(input_ids=x, labels=None, use_cache=False, return_dict=True).logits
        student_logits = student(x)["logits"]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1).mean()
        rows.append(float(kl.item()))
        student.detach_state_()
        student.clear_distillation_caches()

    if not rows:
        return {}
    return {"replacement_logit_kl": float(sum(rows) / len(rows))}


def _copy_single_block_into_wrapper(block_model, wrapper_layer) -> None:
    wrapper_layer.drn_mlp.load_state_dict(block_model.drn.state_dict(), strict=True)
    wrapper_layer.set_drn_scales(block_model.input_scale, block_model.output_scale)
    wrapper_layer.set_drn_output_gain(block_model.output_gain)
    source_resistive = list(block_model.named_resistive_parameters())
    target_resistive = list(wrapper_layer.drn_mlp.named_resistive_parameters())
    if len(source_resistive) != len(target_resistive):
        raise RuntimeError("Single-block and wrapper DRN resistive parameter counts differ.")
    for (source_name, source_tensor), (target_name, target_tensor) in zip(source_resistive, target_resistive):
        if source_tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"Resistive shape mismatch for {source_name} -> {target_name}: "
                f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}."
            )
        target_tensor.copy_(source_tensor.detach())


def _prepare_calibration(
    args: argparse.Namespace,
    teacher: torch.nn.Module,
    loader: DataLoader | None,
    layer_indices: list[int],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any] | None:
    if args.scale_source == "none":
        return None
    if args.calibration is not None:
        return load_calibration(args.calibration)
    if args.activation_cache is not None:
        return calibration_from_cache(args.activation_cache)
    if loader is None:
        return None
    if args.calibration_batches <= 0:
        return None
    payload = calibrate_teacher(
        teacher,
        loader,
        layer_indices,
        device=device,
        max_batches=args.calibration_batches,
        max_quantile_samples=args.max_quantile_samples,
        mlp_input_mode=args.mlp_input_mode,
    )
    payload.update(
        {
            "model_name": args.model_name,
            "layer_indices": layer_indices,
            "scale_source": args.scale_source,
            "mlp_input_mode": args.mlp_input_mode,
        }
    )
    save_calibration(output_dir / "calibration.json", payload)
    return payload


def _layer_scales(args: argparse.Namespace, calibration: dict[str, Any] | None, layer_index: int) -> tuple[float, float]:
    if args.scale_source == "none":
        return 1.0, 1.0
    return scales_from_calibration(calibration, layer_index, source=args.scale_source, default=1.0)


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "default":
        return default_probe_layers(num_layers)
    if raw == "all":
        return list(range(num_layers))
    values: list[int] = []
    for part in (item.strip() for item in raw.split(",") if item.strip()):
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending layer range '{part}'.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    bad = [idx for idx in values if idx < 0 or idx >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}.")
    return sorted(set(values))


def _validate_activation_cache(args: argparse.Namespace, layer_indices: list[int]) -> None:
    metadata = load_cache_metadata(args.activation_cache)
    cache_mode = metadata.get("mlp_input_mode", "normalized")
    if cache_mode != args.mlp_input_mode:
        raise ValueError(
            f"Activation cache was built with mlp_input_mode={cache_mode!r}, "
            f"but this run requested {args.mlp_input_mode!r}."
        )
    missing = [idx for idx in layer_indices if str(idx) not in metadata.get("layers", [])]
    if missing:
        raise ValueError(f"Activation cache is missing requested layer(s): {missing}.")
    for split in ("train", "val"):
        if split not in metadata.get("splits", {}):
            raise ValueError(f"Activation cache is missing split '{split}'.")


def _build_teacher(args: argparse.Namespace) -> torch.nn.Module:
    if not args.debug:
        return load_opt_causal_lm(args.model_name)
    try:
        from transformers import OPTConfig, OPTForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to construct debug OPT teachers.") from exc

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


def _resolved_init_mode(raw: str) -> str:
    if raw == "teacher_mlp":
        return "teacher_frontend_scale"
    return raw


def _teacher_init_scope(raw: str) -> str:
    if raw in {"teacher_mlp", "teacher_frontend", "teacher_frontend_no_scale", "teacher_frontend_scale"}:
        return "fc1_current_frontend_only"
    return "random_drn_native"


def _hard_sigmoid_param(args: argparse.Namespace) -> dict[str, float] | None:
    if args.drn_non_linearity != "hard_sigmoid":
        return None
    return {
        "g_on": float(args.drn_hard_sigmoid_g_on),
        "g_off": float(args.drn_hard_sigmoid_g_off),
        "v_min": float(args.drn_hard_sigmoid_v_min),
        "v_max": float(args.drn_hard_sigmoid_v_max),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--activation_cache", type=Path, default=None)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--activation_cache_open_shards", type=int, default=8)
    parser.add_argument("--activation_cache_workers", type=int, default=0)
    parser.add_argument("--layers", default="default")
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="local_mlp",
    )
    parser.add_argument(
        "--objective_schedule",
        default=None,
        help="Optional staged objective schedule, e.g. 'post_residual:500,rigorous_pretrain:500'.",
    )
    parser.add_argument("--mlp_input_mode", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--alpha_next_ln", type=float, default=0.1)
    parser.add_argument("--alpha_cosine", type=float, default=0.1)
    parser.add_argument("--alpha_norm", type=float, default=0.1)
    parser.add_argument("--alpha_post_residual", type=float, default=0.0)
    parser.add_argument("--alpha_logit_kl", type=float, default=0.0)
    parser.add_argument("--logit_temperature", type=float, default=1.0)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--residual_drift_std", type=float, default=0.0)
    parser.add_argument("--input_noise_mode", choices=["gaussian"], default="gaussian")
    parser.add_argument("--noise_train_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--init_mode",
        choices=[
            "random",
            "teacher_frontend",
            "teacher_frontend_no_scale",
            "teacher_frontend_scale",
            "teacher_mlp",
        ],
        default="random",
    )
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--eval_logit_kl_batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--calibration_batches", type=int, default=16)
    parser.add_argument("--max_quantile_samples", type=int, default=250_000)
    parser.add_argument("--scale_source", choices=["q0_999_abs", "std", "none"], default="q0_999_abs")
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--drn_signed_drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drn_drive_architecture",
        choices=["projected_hidden", "signed_input_free"],
        default="projected_hidden",
    )
    parser.add_argument("--drn_signed_output_weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--drn_non_linearity",
        choices=["perfect_diode", "hard_sigmoid", "linear", "lpw_diode"],
        default="perfect_diode",
    )
    parser.add_argument("--drn_hard_sigmoid_g_on", type=float, default=10.0)
    parser.add_argument("--drn_hard_sigmoid_g_off", type=float, default=1.0e-7)
    parser.add_argument("--drn_hard_sigmoid_v_min", type=float, default=-1.2)
    parser.add_argument("--drn_hard_sigmoid_v_max", type=float, default=1.2)
    parser.add_argument("--drn_learn_drive_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drn_train_current_frontend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drn_hidden_multiplier", type=float, default=None)
    parser.add_argument("--drn_weight_gains", type=float, default=0.1)
    parser.add_argument("--drn_weight_min", type=float, default=1.0e-5)
    parser.add_argument("--drn_weight_max", type=float, default=None)
    parser.add_argument("--drn_bias_gain", type=float, default=0.0)
    parser.add_argument("--drn_init_drive_scale", type=float, default=1.0)
    parser.add_argument("--drn_voltage_amp", type=float, default=1.0)
    parser.add_argument("--drn_current_amp", type=float, default=1.0)
    parser.add_argument("--drn_learn_amplification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drn_amp_lr", type=float, default=None)
    parser.add_argument("--output_gain_lr", type=float, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be positive.")
    if args.eval_iters <= 0:
        raise ValueError("--eval_iters must be positive.")
    if args.eval_logit_kl_batches < 0:
        raise ValueError("--eval_logit_kl_batches must be non-negative.")
    args._parsed_objective_schedule = _parse_objective_schedule(args.objective_schedule, args.steps)
    for name in ("alpha_next_ln", "alpha_cosine", "alpha_norm", "alpha_post_residual", "alpha_logit_kl"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.logit_temperature <= 0.0:
        raise ValueError("--logit_temperature must be positive.")
    if args.input_noise_std < 0.0:
        raise ValueError("--input_noise_std must be non-negative.")
    if args.residual_drift_std < 0.0:
        raise ValueError("--residual_drift_std must be non-negative.")
    if args.activation_cache_open_shards <= 0:
        raise ValueError("--activation_cache_open_shards must be positive.")
    if args.activation_cache_workers < 0:
        raise ValueError("--activation_cache_workers must be non-negative.")
    if args.drn_voltage_amp <= 0.0:
        raise ValueError("--drn_voltage_amp must be positive.")
    if args.drn_current_amp <= 0.0:
        raise ValueError("--drn_current_amp must be positive.")
    if args.drn_amp_lr is not None and args.drn_amp_lr <= 0.0:
        raise ValueError("--drn_amp_lr must be positive when provided.")
    if args.output_gain_lr is not None and args.output_gain_lr <= 0.0:
        raise ValueError("--output_gain_lr must be positive when provided.")
    return args


def _parse_objective_schedule(raw: str | None, total_steps: int) -> list[tuple[str, int]] | None:
    if raw is None:
        return None
    schedule: list[tuple[str, int]] = []
    running_steps = 0
    for item in (part.strip() for part in raw.split(",") if part.strip()):
        if ":" not in item:
            raise ValueError(f"Invalid objective schedule item '{item}'. Expected objective:steps.")
        objective, steps_raw = item.split(":", 1)
        objective = objective.strip()
        if objective not in OBJECTIVES:
            raise ValueError(f"Unsupported schedule objective '{objective}'.")
        steps = int(steps_raw)
        if steps <= 0:
            raise ValueError("Objective schedule steps must be positive.")
        running_steps += steps
        schedule.append((objective, running_steps))
    if running_steps != int(total_steps):
        raise ValueError(f"Objective schedule sums to {running_steps} steps, expected --steps={total_steps}.")
    return schedule


def _objective_for_step(args: argparse.Namespace, step: int) -> str:
    schedule = getattr(args, "_parsed_objective_schedule", None)
    if not schedule:
        return args.objective
    for objective, end_step in schedule:
        if int(step) <= end_step:
            return objective
    return schedule[-1][0]


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


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row))
        for key in keys
    }


def _count_tensors(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
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
