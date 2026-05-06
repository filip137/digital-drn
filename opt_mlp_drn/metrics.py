from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    denom = left_norm * right_norm
    if float(denom.item()) <= 1.0e-12:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).div(denom).item())


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    target_energy = torch.mean(target.detach().float() ** 2).clamp_min(1.0e-12)
    mse = F.mse_loss(pred.detach().float(), target.detach().float())
    return float((mse / target_energy).item())


def relative_rms_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    denom = torch.sum(target_flat * target_flat)
    if float(denom.item()) <= 1.0e-12:
        return float("nan")
    diff = pred_flat - target_flat
    return float(torch.sqrt(torch.sum(diff * diff) / denom).item())


def q_abs_error(pred: torch.Tensor, target: torch.Tensor, q: float = 0.99) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1].")
    values = (pred.detach().float() - target.detach().float()).abs().reshape(-1)
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, q).item())


def grad_global_norm(tensors: list[torch.Tensor]) -> float:
    total_sq = 0.0
    for tensor in tensors:
        grad = tensor.grad
        if grad is None:
            continue
        grad_values = grad.detach().float()
        total_sq += float(torch.sum(grad_values * grad_values).item())
    return math.sqrt(total_sq)


def logit_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = max(float(temperature), 1.0e-6)
    teacher_log_probs = F.log_softmax(teacher_logits / temp, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    return (temp * temp) * torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1).mean()


def shifted_causal_logit_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = max(float(temperature), 1.0e-6)
    teacher = teacher_logits.float()[:, :-1, :]
    student = student_logits.float()[:, :-1, :]
    if attention_mask is None:
        mask = torch.ones(student.shape[:2], dtype=student.dtype, device=student.device)
    else:
        mask = attention_mask[:, 1:].to(device=student.device, dtype=student.dtype)

    teacher_log_probs = F.log_softmax(teacher / temp, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student / temp, dim=-1)
    kl_per_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    loss = (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)
    return loss * (temp * temp)


def hidden_drift_metrics(
    student_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    teacher_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    max_depth: int | None = None,
    prefix: str = "hidden",
) -> dict[str, float]:
    count = min(len(student_hidden_states), len(teacher_hidden_states))
    if max_depth is not None:
        count = min(count, int(max_depth) + 1)
    rows: dict[str, float] = {}
    for depth in range(1, count):
        student = student_hidden_states[depth]
        teacher = teacher_hidden_states[depth]
        mse = F.mse_loss(student, teacher.detach())
        rows[f"{prefix}_{depth}_mse"] = float(mse.detach().item())
        rows[f"{prefix}_{depth}_rel_mse"] = relative_mse(student, teacher)
        rows[f"{prefix}_{depth}_rel_rms"] = relative_rms_error(student, teacher)
    if count > 1:
        final_depth = count - 1
        rows[f"{prefix}_max_rel_rms"] = max(rows[f"{prefix}_{depth}_rel_rms"] for depth in range(1, count))
        rows[f"{prefix}_final_rel_rms"] = rows[f"{prefix}_{final_depth}_rel_rms"]
    return rows


def drn_saturation_fraction(module: Any, *, eps: float = 1.0e-6) -> float:
    """Estimate the fraction of nonlinear DRN state entries at a diode boundary.

    The exact meaning is nonlinearity-dependent: for ``perfect_diode`` this is
    the fraction of sign-split hidden entries pinned at the inactive boundary;
    for ``hard_sigmoid`` this is the fraction outside the configured linear
    voltage window. Other nonlinearities currently return NaN because they do
    not expose a simple saturation boundary.
    """

    drn_mlp = getattr(module, "drn_mlp", None)
    if drn_mlp is None:
        drn_mlp = getattr(module, "drn", module)
    block = getattr(drn_mlp, "block", None)
    energy = getattr(block, "energy", None)
    if energy is None:
        return float("nan")

    non_linearity = str(getattr(energy, "_non_linearity", ""))
    layers = list(getattr(energy, "free_layers")())
    if len(layers) <= 1:
        return float("nan")
    nonlinear_layers = layers[:-1]

    saturated = 0
    total = 0
    for layer in nonlinear_layers:
        state = layer.state.detach().float()
        if state.numel() == 0:
            continue
        total += state.numel()
        if non_linearity == "perfect_diode":
            if state.size(-1) % 2 != 0:
                continue
            half = state.size(-1) // 2
            excitatory = state[..., :half]
            inhibitory = state[..., half:]
            saturated += int((excitatory <= eps).sum().item())
            saturated += int((inhibitory >= -eps).sum().item())
        elif non_linearity == "hard_sigmoid":
            params = getattr(energy, "_hard_sigmoid_param", {})
            if "v_min" not in params or "v_max" not in params:
                return float("nan")
            v_min = float(params["v_min"])
            v_max = float(params["v_max"])
            saturated += int(((state <= v_min + eps) | (state >= v_max - eps)).sum().item())
        else:
            return float("nan")
    if total == 0:
        return float("nan")
    return float(saturated / total)


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int))})
    return {
        key: float(sum(float(row[key]) for row in rows if key in row) / sum(1 for row in rows if key in row))
        for key in keys
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
