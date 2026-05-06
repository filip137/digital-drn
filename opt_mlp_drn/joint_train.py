from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoints import (
    load_single_block_checkpoint_dir,
    load_single_block_checkpoint_into_layer,
    trainable_tensors,
)
from .data import load_explicit_text_datasets, load_text_datasets
from .metrics import append_jsonl, grad_global_norm, hidden_drift_metrics, logit_kl, save_json, shifted_causal_logit_kl
from .model import OPTMLPDRNForCausalLM, load_opt_causal_lm, parse_layer_indices
from .teacher import apply_next_layer_norm


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = "char" if args.debug and args.tokenizer == "auto" else args.tokenizer
    train_dataset, val_dataset, test_dataset, _encode, _decode = _load_datasets(args, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if test_dataset is not None
        else None
    )
    train_iter = _cycle(train_loader)

    teacher = _build_teacher(args).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    model = _build_student(args, teacher).to(device)
    lm_head_untied = False
    if _requires_separate_trainable_lm_head(args):
        lm_head_untied = _untie_lm_head_if_tied(model)
    if args.checkpoint_dir is not None:
        load_single_block_checkpoint_dir(model, args.checkpoint_dir)
    if args.checkpoint_path:
        _load_checkpoint_paths(model, args.checkpoint_path)
    if args.resume_checkpoint is not None:
        _load_joint_checkpoint(model, args.resume_checkpoint)
    _set_trainable_scope(
        model,
        args.trainable_scope,
        train_lm_head=args.train_lm_head,
        train_final_ln=args.train_final_ln,
    )
    params = trainable_tensors(model)
    if not params:
        raise RuntimeError("No trainable DRN tensors found.")
    optimizer = _make_optimizer(model, args)

    output_dir = Path(args.output_dir) / f"opt_mlp_drn_joint_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "model_name": args.model_name,
        "experiment": "full_transformer_joint_distillation",
        "replace_mlp_layers": args.replace_mlp_layers,
        "replaced_layer_indices": model.replaced_layer_indices,
        "hidden_weight": args.hidden_weight,
        "hidden_layers": args.hidden_layers,
        "hidden_loss_type": args.hidden_loss_type,
        "kl_weight": args.kl_weight,
        "kl_temperature": args.kl_temperature,
        "ce_weight": args.ce_weight,
        "post_residual_weight": args.post_residual_weight,
        "next_ln_weight": args.next_ln_weight,
        "delta_cosine_weight": args.delta_cosine_weight,
        "delta_norm_weight": args.delta_norm_weight,
        "replacement_schedule": args.replacement_schedule,
        "distill_objective": args.distill_objective,
        "distill_beta": args.distill_beta,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "trainable_scope": args.trainable_scope,
        "train_lm_head": args.train_lm_head,
        "train_final_ln": args.train_final_ln,
        "lm_head_untied_from_embeddings": lm_head_untied,
        "trainable_embedding_params": _num_trainable_embedding_params(model),
        "trainable_lm_head_params": _num_trainable_lm_head_params(model),
        "drn_signed_drive": args.drn_signed_drive,
        "drn_drive_architecture": args.drn_drive_architecture,
        "lr": args.lr,
        "attn_lr": args.attn_lr,
        "ln_lr": args.ln_lr,
        "drn_amp_lr": args.drn_amp_lr,
        "trainable_params": sum(t.numel() for t in params),
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "checkpoint_paths_init": list(args.checkpoint_path or []),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "train_data": str(args.train_data) if args.train_data else None,
        "val_data": str(args.val_data) if args.val_data else None,
        "test_data": str(args.test_data) if args.test_data else None,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "run_metadata.json", metadata)

    metrics_path = output_dir / "metrics.jsonl"
    initial = _evaluate(model, teacher, val_loader, device, args)
    best_loss = initial["loss"]
    best_metric = _selection_metric(initial, args.early_stopping_metric)
    best_step = 0
    bad_evals = 0
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    _save_checkpoint(best_checkpoint_path, model, optimizer, 0, args, initial)
    append_jsonl(metrics_path, {"stage": "eval", "step": 0, **initial})
    print({"stage": "eval", "step": 0, **initial})
    last_grad_norm = 0.0
    stopped_early = False
    final_step = 0
    for step in range(1, args.steps + 1):
        final_step = step
        model.train()
        replacement_probability = _replacement_probability_for_step(step, args.steps, args.replacement_schedule)
        model.set_replacement_probability(replacement_probability)
        x, y = next(train_iter)
        loss, _metrics = _joint_loss(model, teacher, x.to(device), y.to(device), args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = grad_global_norm(params)
        if args.grad_clip > 0.0:
            clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        model.clamp_resistive_params_()
        model.detach_state_()
        model.clear_distillation_caches()
        append_jsonl(
            metrics_path,
            {
                "stage": "train",
                "step": step,
                "loss": float(loss.detach().item()),
                "grad_norm": last_grad_norm,
                "replacement_probability": replacement_probability,
                **_metrics,
            },
        )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = _evaluate(model, teacher, val_loader, device, args)
            best_loss = min(best_loss, metrics["loss"])
            current_metric = _selection_metric(metrics, args.early_stopping_metric)
            if current_metric < best_metric - float(args.min_delta):
                best_metric = current_metric
                best_step = step
                bad_evals = 0
                _save_checkpoint(best_checkpoint_path, model, optimizer, step, args, metrics)
            else:
                bad_evals += 1
            append_jsonl(metrics_path, {"stage": "eval", "step": step, **metrics})
            print({"stage": "eval", "step": step, **metrics})
            if args.patience > 0 and bad_evals >= args.patience:
                stopped_early = True
                break

    final = _evaluate(model, teacher, val_loader, device, args)
    checkpoint_path = output_dir / "checkpoint_last.pt"
    test_metrics = None
    final.update(
        {
            "initial_loss": initial["loss"],
            "best_loss": best_loss,
            "best_step": best_step,
            "best_validation_metric": best_metric,
            "early_stopping_metric": args.early_stopping_metric,
            "stopped_early": stopped_early,
            "final_step": final_step,
            "last_train_grad_norm": last_grad_norm,
            "peak_memory_mb": _cuda_peak_memory_mb(),
        }
    )
    _save_checkpoint(checkpoint_path, model, optimizer, final_step, args, final)
    if test_loader is not None:
        _load_joint_checkpoint(model, best_checkpoint_path)
        test_metrics = _evaluate(model, teacher, test_loader, device, args)
        append_jsonl(metrics_path, {"stage": "test", "step": best_step, **test_metrics})
        final.update({f"test_{key}": value for key, value in test_metrics.items()})
    final["checkpoint_paths"] = {"last": str(checkpoint_path), "best": str(best_checkpoint_path)}
    save_json(output_dir / "final_metrics.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


@torch.no_grad()
def _evaluate(model, teacher, loader, device: torch.device, args) -> dict[str, float]:
    was_training = model.training
    previous_probability = _current_replacement_probability(model)
    model.eval()
    model.set_replacement_probability(1.0)
    rows = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.eval_iters:
            break
        loss, metrics = _joint_loss(model, teacher, x.to(device), y.to(device), args)
        rows.append({"loss": float(loss.detach().item()), **metrics})
        model.detach_state_()
        model.clear_distillation_caches()
    if was_training:
        model.train()
    model.set_replacement_probability(previous_probability)
    return _mean_rows(rows)


def _joint_loss(model, teacher, input_ids: torch.Tensor, targets: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        teacher_out = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        teacher_hidden = list(teacher_out.hidden_states)
        teacher_logits = teacher_out.logits
    student_out = model(input_ids, return_hidden_states=True, attention_mask=attention_mask)
    student_hidden = student_out["hidden_states"]
    if student_hidden is None:
        raise RuntimeError("Student did not return hidden states.")
    hidden_depths = _selected_hidden_depths(
        args.hidden_layers,
        model.replaced_layer_indices,
        max_depth=min(len(student_hidden), len(teacher_hidden)) - 1,
    )
    hidden_loss = _hidden_match_loss(
        student_hidden,
        teacher_hidden,
        hidden_depths,
        loss_type=args.hidden_loss_type,
        device=input_ids.device,
    )
    replacement_losses = _replacement_auxiliary_losses(model, teacher, input_ids.device)
    unshifted_kl = logit_kl(teacher_logits, student_out["logits"], temperature=args.kl_temperature)
    distill_kl = shifted_causal_logit_kl(
        teacher_logits,
        student_out["logits"],
        attention_mask=attention_mask,
        temperature=args.kl_temperature,
    )
    student_ce = F.cross_entropy(student_out["logits"].reshape(-1, student_out["logits"].size(-1)), targets.reshape(-1))
    teacher_ce = F.cross_entropy(teacher_logits.reshape(-1, teacher_logits.size(-1)), targets.reshape(-1))
    shifted_student_ce = F.cross_entropy(
        student_out["logits"][:, :-1, :].reshape(-1, student_out["logits"].size(-1)),
        input_ids[:, 1:].reshape(-1),
    )
    shifted_teacher_ce = F.cross_entropy(
        teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )
    if args.distill_objective == "logit_kl":
        beta = float(args.distill_beta)
        loss = beta * distill_kl + (1.0 - beta) * shifted_student_ce
    elif args.distill_objective == "hidden_kl":
        loss = float(args.hidden_weight) * hidden_loss + float(args.kl_weight) * distill_kl
        if args.ce_weight > 0.0:
            loss = loss + float(args.ce_weight) * shifted_student_ce
    elif args.distill_objective == "progressive_hidden_kl":
        loss = float(args.kl_weight) * distill_kl
        loss = loss + float(args.hidden_weight) * hidden_loss
        loss = loss + float(args.post_residual_weight) * replacement_losses["post_residual_mse"]
        loss = loss + float(args.next_ln_weight) * replacement_losses["next_ln_mse"]
        loss = loss + float(args.delta_cosine_weight) * replacement_losses["delta_cosine_loss"]
        loss = loss + float(args.delta_norm_weight) * replacement_losses["delta_norm_loss"]
        if args.ce_weight > 0.0:
            loss = loss + float(args.ce_weight) * shifted_student_ce
    else:
        raise ValueError(f"Unsupported distill_objective '{args.distill_objective}'.")
    metrics = {
        "hidden_loss": float(hidden_loss.detach().item()),
        "hidden_depths_count": len(hidden_depths),
        "logit_kl": float(distill_kl.detach().item()),
        "unshifted_logit_kl": float(unshifted_kl.detach().item()),
        "distill_kl_loss": float(distill_kl.detach().item()),
        "post_residual_mse": float(replacement_losses["post_residual_mse"].detach().item()),
        "next_ln_mse": float(replacement_losses["next_ln_mse"].detach().item()),
        "delta_mse": float(replacement_losses["delta_mse"].detach().item()),
        "delta_cosine_loss": float(replacement_losses["delta_cosine_loss"].detach().item()),
        "delta_norm_loss": float(replacement_losses["delta_norm_loss"].detach().item()),
        "mean_delta_cosine": float(replacement_losses["mean_delta_cosine"].detach().item()),
        "mean_delta_norm_ratio": float(replacement_losses["mean_delta_norm_ratio"].detach().item()),
        "replacement_probability": float(replacement_losses["replacement_probability"].detach().item()),
        "student_replacement_fraction": float(replacement_losses["student_replacement_fraction"].detach().item()),
        "student_ce_loss": float(student_ce.detach().item()),
        "teacher_ce_loss": float(teacher_ce.detach().item()),
        "shifted_student_ce_loss": float(shifted_student_ce.detach().item()),
        "shifted_teacher_ce_loss": float(shifted_teacher_ce.detach().item()),
        "ce_loss": float(shifted_student_ce.detach().item()),
        "teacher_student_ce_gap": float((student_ce.detach() - teacher_ce.detach()).item()),
        "shifted_teacher_student_ce_gap": float((shifted_student_ce.detach() - shifted_teacher_ce.detach()).item()),
    }
    student_ce_value = float(shifted_student_ce.detach().item())
    teacher_ce_value = float(shifted_teacher_ce.detach().item())
    if student_ce_value < 50.0:
        metrics["student_ppl"] = float(math.exp(student_ce_value))
        metrics["ppl"] = metrics["student_ppl"]
    if teacher_ce_value < 50.0:
        metrics["teacher_ppl"] = float(math.exp(teacher_ce_value))
    metrics.update(hidden_drift_metrics(student_hidden, teacher_hidden, prefix="hidden"))
    return loss, metrics


def _selected_hidden_depths(
    raw: str,
    replaced_layer_indices: list[int],
    *,
    max_depth: int,
) -> list[int]:
    if max_depth <= 0:
        return []
    text = str(raw).strip().lower()
    if text in {"all", ""}:
        return list(range(1, max_depth + 1))
    if text in {"replaced", "replaced_outputs", "replacement_outputs"}:
        return [depth for depth in (index + 1 for index in replaced_layer_indices) if 1 <= depth <= max_depth]
    return [depth for depth in parse_layer_indices(text, max_depth + 1) if depth > 0]


def _hidden_match_loss(
    student_hidden: list[torch.Tensor],
    teacher_hidden: list[torch.Tensor],
    depths: list[int],
    *,
    loss_type: str,
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for depth in depths:
        if depth >= len(student_hidden) or depth >= len(teacher_hidden):
            continue
        student = student_hidden[depth]
        teacher = teacher_hidden[depth].detach()
        if loss_type == "mse":
            losses.append(F.mse_loss(student, teacher))
        elif loss_type == "normed_mse":
            losses.append(F.mse_loss(_unit_normalize(student), _unit_normalize(teacher)))
        elif loss_type == "cosine":
            losses.append(_token_cosine_loss(student, teacher))
        else:
            raise ValueError(f"Unsupported hidden_loss_type '{loss_type}'.")
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def _replacement_auxiliary_losses(model: OPTMLPDRNForCausalLM, teacher, device: torch.device) -> dict[str, torch.Tensor]:
    delta_losses = []
    post_losses = []
    next_ln_losses = []
    cosine_losses = []
    norm_losses = []
    cosines = []
    norm_ratios = []
    probabilities = []
    used_student = []
    for layer in model.replaced_layers():
        cache = layer.distillation_cache()
        student_delta = cache.student_delta
        teacher_delta = cache.teacher_delta.detach()
        delta_losses.append(F.mse_loss(student_delta, teacher_delta))
        post_losses.append(F.mse_loss(cache.student_post_residual, cache.teacher_post_residual.detach()))
        next_ln_losses.append(
            F.mse_loss(
                apply_next_layer_norm(teacher, layer.layer_index, cache.student_post_residual),
                apply_next_layer_norm(teacher, layer.layer_index, cache.teacher_post_residual.detach()).detach(),
            )
        )
        cosine_loss = _token_cosine_loss(student_delta, teacher_delta)
        norm_ratio, norm_loss = _norm_ratio_loss(student_delta, teacher_delta)
        cosine_losses.append(cosine_loss)
        norm_losses.append(norm_loss)
        cosines.append(1.0 - cosine_loss.detach())
        norm_ratios.append(norm_ratio.detach())
        probabilities.append(torch.tensor(cache.replacement_probability, device=device))
        used_student.append(torch.tensor(float(cache.used_student_mlp), device=device))

    return {
        "delta_mse": _mean_or_zero(delta_losses, device),
        "post_residual_mse": _mean_or_zero(post_losses, device),
        "next_ln_mse": _mean_or_zero(next_ln_losses, device),
        "delta_cosine_loss": _mean_or_zero(cosine_losses, device),
        "delta_norm_loss": _mean_or_zero(norm_losses, device),
        "mean_delta_cosine": _mean_or_zero(cosines, device),
        "mean_delta_norm_ratio": _mean_or_zero(norm_ratios, device),
        "replacement_probability": _mean_or_zero(probabilities, device),
        "student_replacement_fraction": _mean_or_zero(used_student, device),
    }


def _unit_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), p=2, dim=-1, eps=1.0e-12)


def _token_cosine_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(left.float(), right.detach().float(), dim=-1, eps=1.0e-12)
    return 1.0 - cosine.mean()


def _norm_ratio_loss(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left.float().reshape(-1))
    right_norm = torch.linalg.vector_norm(right.detach().float().reshape(-1)).clamp_min(1.0e-12)
    ratio = left_norm / right_norm
    return ratio, torch.log(ratio.clamp_min(1.0e-12)).pow(2)


def _mean_or_zero(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device)
    return torch.stack([value.to(device) for value in values]).mean()


def _replacement_probability_for_step(step: int, total_steps: int, raw_schedule: str) -> float:
    values = _parse_replacement_schedule(raw_schedule)
    if len(values) == 1:
        return values[0]
    index = min(len(values) - 1, max(0, int((int(step) - 1) * len(values) / max(1, int(total_steps)))))
    return values[index]


def _parse_replacement_schedule(raw_schedule: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw_schedule).split(",") if part.strip()]
    if not values:
        raise ValueError("--replacement_schedule must contain at least one probability.")
    bad = [value for value in values if value < 0.0 or value > 1.0]
    if bad:
        raise ValueError(f"--replacement_schedule probabilities must lie in [0, 1], got {bad}.")
    return values


def _current_replacement_probability(model: OPTMLPDRNForCausalLM) -> float:
    layers = model.replaced_layers()
    if not layers:
        return 1.0
    return float(layers[0].replacement_probability)


def _make_optimizer(model: OPTMLPDRNForCausalLM, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_params(params, *, lr: float, weight_decay: float, label: str) -> None:
        selected = []
        for tensor in params:
            if not tensor.requires_grad:
                continue
            key = id(tensor)
            if key in seen:
                continue
            seen.add(key)
            selected.append(tensor)
        if selected:
            groups.append({"params": selected, "lr": float(lr), "weight_decay": float(weight_decay), "name": label})

    if args.drn_amp_lr is None:
        for layer in model.replaced_layers():
            add_params(
                list(layer.drn_mlp.parameters()) + layer.drn_mlp.resistive_param_states(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                label=f"layer_{layer.layer_index}/drn",
            )
    else:
        for layer in model.replaced_layers():
            layer.drn_mlp.block.amp_learning_rate = float(args.drn_amp_lr)
            for group in layer.drn_mlp.optimizer_param_groups():
                params = []
                for tensor in group.get("params", []):
                    if not tensor.requires_grad:
                        continue
                    key = id(tensor)
                    if key in seen:
                        continue
                    seen.add(key)
                    params.append(tensor)
                if not params:
                    continue
                normalized = dict(group)
                normalized["params"] = params
                normalized.setdefault("name", f"layer_{layer.layer_index}/drn")
                groups.append(normalized)

    for layer in model.replaced_layers():
        add_params(
            _attention_output_parameters(layer),
            lr=args.attn_lr,
            weight_decay=args.weight_decay,
            label=f"layer_{layer.layer_index}/attn_out",
        )
        add_params(
            _attention_qkv_parameters(layer),
            lr=args.attn_lr,
            weight_decay=args.weight_decay,
            label=f"layer_{layer.layer_index}/attn_qkv",
        )
        add_params(
            _layer_norm_parameters(layer),
            lr=args.ln_lr,
            weight_decay=0.0,
            label=f"layer_{layer.layer_index}/layer_norms",
        )

    add_params(
        [tensor for tensor in trainable_tensors(model) if id(tensor) not in seen],
        lr=args.lr,
        weight_decay=args.weight_decay,
        label="other",
    )

    if not groups:
        groups = [{"params": trainable_tensors(model), "lr": args.lr, "weight_decay": args.weight_decay}]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def _set_trainable_scope(
    model: OPTMLPDRNForCausalLM,
    scope: str,
    *,
    train_lm_head: bool = False,
    train_final_ln: bool = False,
) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for tensor in model.resistive_param_states():
        tensor.requires_grad_(False)

    if scope == "all_student":
        for param in model.parameters():
            param.requires_grad_(True)
        for tensor in model.resistive_param_states():
            tensor.requires_grad_(True)
        return

    for layer in model.replaced_layers():
        for param in layer.drn_mlp.parameters():
            param.requires_grad_(True)
        for tensor in layer.drn_mlp.resistive_param_states():
            tensor.requires_grad_(True)

        if scope in {"drn_attn_out", "drn_attn_full_ln", "drn_attn_full_ln_final_lm"}:
            for param in _attention_output_parameters(layer):
                param.requires_grad_(True)
        if scope in {"drn_attn_full_ln", "drn_attn_full_ln_final_lm"}:
            for param in _attention_qkv_parameters(layer):
                param.requires_grad_(True)
            for param in _layer_norm_parameters(layer):
                param.requires_grad_(True)

    if train_lm_head or scope == "drn_attn_full_ln_final_lm":
        for param in _lm_head_parameters(model):
            param.requires_grad_(True)
    if train_final_ln or scope == "drn_attn_full_ln_final_lm":
        for param in _final_layer_norm_parameters(model):
            param.requires_grad_(True)


def _requires_separate_trainable_lm_head(args: argparse.Namespace) -> bool:
    # OPT ties lm_head.weight to decoder.embed_tokens.weight by default. For scopes that
    # train the output projection while keeping the frozen backbone, replace lm_head with
    # an independent copy before setting requires_grad flags.
    return args.trainable_scope != "all_student" and (
        bool(args.train_lm_head) or args.trainable_scope == "drn_attn_full_ln_final_lm"
    )


def _untie_lm_head_if_tied(model: OPTMLPDRNForCausalLM) -> bool:
    lm_head = getattr(model.base, "lm_head", None)
    embed_tokens = _token_embedding_module(model)
    if lm_head is None or embed_tokens is None:
        return False
    if not isinstance(lm_head, nn.Linear):
        return False
    lm_weight = lm_head.weight
    embed_weight = embed_tokens.weight
    if lm_weight.untyped_storage().data_ptr() != embed_weight.untyped_storage().data_ptr():
        return False

    replacement = nn.Linear(
        lm_head.in_features,
        lm_head.out_features,
        bias=lm_head.bias is not None,
        device=lm_weight.device,
        dtype=lm_weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(lm_weight.detach())
        if lm_head.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(lm_head.bias.detach())
    model.base.lm_head = replacement
    if hasattr(model.base, "config"):
        model.base.config.tie_word_embeddings = False
    return True


def _attention_output_parameters(layer) -> list[torch.Tensor]:
    return list(layer.self_attn.out_proj.parameters())


def _attention_qkv_parameters(layer) -> list[torch.Tensor]:
    return (
        list(layer.self_attn.q_proj.parameters())
        + list(layer.self_attn.k_proj.parameters())
        + list(layer.self_attn.v_proj.parameters())
    )


def _layer_norm_parameters(layer) -> list[torch.Tensor]:
    return list(layer.self_attn_layer_norm.parameters()) + list(layer.final_layer_norm.parameters())


def _lm_head_parameters(model: OPTMLPDRNForCausalLM) -> list[torch.Tensor]:
    lm_head = getattr(model.base, "lm_head", None)
    if lm_head is None:
        return []
    return list(lm_head.parameters())


def _num_trainable_lm_head_params(model: OPTMLPDRNForCausalLM) -> int:
    return sum(param.numel() for param in _lm_head_parameters(model) if param.requires_grad)


def _token_embedding_module(model: OPTMLPDRNForCausalLM):
    decoder = getattr(getattr(model.base, "model", None), "decoder", None)
    if decoder is None:
        return None
    return getattr(decoder, "embed_tokens", None)


def _num_trainable_embedding_params(model: OPTMLPDRNForCausalLM) -> int:
    embed_tokens = _token_embedding_module(model)
    if embed_tokens is None:
        return 0
    return sum(param.numel() for param in embed_tokens.parameters() if param.requires_grad)


def _final_layer_norm_parameters(model: OPTMLPDRNForCausalLM) -> list[torch.Tensor]:
    final_ln = getattr(model.base.model.decoder, "final_layer_norm", None)
    if final_ln is None:
        return []
    return list(final_ln.parameters())


def _load_checkpoint_paths(model: OPTMLPDRNForCausalLM, checkpoint_paths: list[str]) -> None:
    layers = {layer.layer_index: layer for layer in model.replaced_layers()}
    for raw in checkpoint_paths:
        if "=" not in raw:
            raise ValueError("--checkpoint_path entries must have the form LAYER=PATH.")
        layer_raw, path_raw = raw.split("=", 1)
        layer_index = int(layer_raw)
        if layer_index not in layers:
            raise ValueError(
                f"Checkpoint provided for layer {layer_index}, but replaced layers are {sorted(layers)}."
            )
        load_single_block_checkpoint_into_layer(path_raw, layers[layer_index])


def _load_joint_checkpoint(model: OPTMLPDRNForCausalLM, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Could not load joint checkpoint {checkpoint_path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    resistive = checkpoint.get("resistive_parameters", {})
    if resistive:
        targets = dict(model.named_resistive_parameters())
        missing_resistive = sorted(set(targets) - set(resistive))
        unexpected_resistive = sorted(set(resistive) - set(targets))
        if missing_resistive or unexpected_resistive:
            raise RuntimeError(
                f"Could not load resistive tensors from {checkpoint_path}: "
                f"missing={missing_resistive}, unexpected={unexpected_resistive}"
            )
        with torch.no_grad():
            for name, value in resistive.items():
                target = targets[name]
                target.copy_(value.to(device=target.device, dtype=target.dtype))


def _build_student(args, teacher) -> OPTMLPDRNForCausalLM:
    return OPTMLPDRNForCausalLM(
        copy.deepcopy(teacher).cpu(),
        replace_mlp_layers=args.replace_mlp_layers,
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


def _load_datasets(args: argparse.Namespace, tokenizer: str):
    if args.train_data is None and args.val_data is None and args.test_data is None:
        train_dataset, val_dataset, encode, decode = load_text_datasets(
            args.data,
            model_name=args.model_name,
            tokenizer=tokenizer,
            train_frac=args.train_frac,
            block_size=args.block_size,
            vocab_size=128 if args.debug else None,
        )
        return train_dataset, val_dataset, None, encode, decode
    if args.train_data is None or args.val_data is None:
        raise ValueError("--train_data and --val_data must be provided together for explicit splits.")
    return load_explicit_text_datasets(
        args.train_data,
        args.val_data,
        args.test_data,
        model_name=args.model_name,
        tokenizer=tokenizer,
        block_size=args.block_size,
        vocab_size=128 if args.debug else None,
    )


def _selection_metric(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Validation metric '{metric_name}' was not produced. Available: {sorted(metrics)}")
    return float(metrics[metric_name])


def _save_checkpoint(
    path: Path,
    model: OPTMLPDRNForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": metrics,
            "resistive_parameters": {
                name: tensor.detach().cpu() for name, tensor in model.named_resistive_parameters()
            },
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train_data", type=Path, default=None)
    parser.add_argument("--val_data", type=Path, default=None)
    parser.add_argument("--test_data", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_path",
        action="append",
        default=[],
        help="Single-block initializer in the form LAYER=/path/to/checkpoint_last.pt. May be repeated.",
    )
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--replace_mlp_layers", default="all")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument(
        "--early_stopping_metric",
        choices=[
            "loss",
            "logit_kl",
            "distill_kl_loss",
            "hidden_final_rel_rms",
            "hidden_max_rel_rms",
            "post_residual_mse",
            "next_ln_mse",
            "delta_mse",
            "delta_cosine_loss",
            "delta_norm_loss",
            "student_ce_loss",
            "shifted_student_ce_loss",
            "ppl",
        ],
        default="logit_kl",
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--hidden_weight", type=float, default=1.0)
    parser.add_argument(
        "--hidden_layers",
        default="all",
        help="Hidden-state depths to match: all, replaced_outputs, or an index/range string such as 10-12.",
    )
    parser.add_argument("--hidden_loss_type", choices=["mse", "normed_mse", "cosine"], default="mse")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=0.0)
    parser.add_argument("--post_residual_weight", type=float, default=0.0)
    parser.add_argument("--next_ln_weight", type=float, default=0.0)
    parser.add_argument("--delta_cosine_weight", type=float, default=0.0)
    parser.add_argument("--delta_norm_weight", type=float, default=0.0)
    parser.add_argument(
        "--replacement_schedule",
        default="1.0",
        help="Comma-separated DRN replacement probabilities used across equal training segments, e.g. 0.1,0.3,0.6,1.0.",
    )
    parser.add_argument("--distill_objective", choices=["hidden_kl", "logit_kl", "progressive_hidden_kl"], default="hidden_kl")
    parser.add_argument("--distill_beta", type=float, default=1.0)
    parser.add_argument(
        "--trainable_scope",
        choices=[
            "drn_only",
            "drn_attn_out",
            "drn_attn_full_ln",
            "drn_attn_full_ln_final_lm",
            "all_student",
        ],
        default="drn_only",
    )
    parser.add_argument("--train_lm_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_final_ln", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--attn_lr", type=float, default=1.0e-5)
    parser.add_argument("--ln_lr", type=float, default=1.0e-5)
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
    parser.add_argument("--drn_amp_lr", type=float, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    parse_layer_indices(args.replace_mlp_layers, 3 if args.debug else 12)
    if args.steps <= 0 or args.eval_interval <= 0 or args.eval_iters <= 0:
        raise ValueError("step and eval counts must be positive.")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative.")
    if args.min_delta < 0.0:
        raise ValueError("--min_delta must be non-negative.")
    for name in (
        "hidden_weight",
        "kl_weight",
        "ce_weight",
        "post_residual_weight",
        "next_ln_weight",
        "delta_cosine_weight",
        "delta_norm_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.kl_temperature <= 0.0:
        raise ValueError("--kl_temperature must be positive.")
    if not 0.0 <= args.distill_beta <= 1.0:
        raise ValueError("--distill_beta must lie in [0, 1].")
    _parse_replacement_schedule(args.replacement_schedule)
    max_hidden_depth = 3 if args.debug else 12
    _selected_hidden_depths(args.hidden_layers, parse_layer_indices(args.replace_mlp_layers, max_hidden_depth), max_depth=max_hidden_depth)
    if args.test_data is not None and (args.train_data is None or args.val_data is None):
        raise ValueError("--test_data requires explicit --train_data and --val_data.")
    if args.drn_amp_lr is not None and args.drn_amp_lr <= 0.0:
        raise ValueError("--drn_amp_lr must be positive when provided.")
    if args.attn_lr <= 0.0 or args.ln_lr <= 0.0:
        raise ValueError("--attn_lr and --ln_lr must be positive.")
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
