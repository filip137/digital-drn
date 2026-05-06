from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drn_architecture import (
    build_tokenwise_drn_mlp,
    resolve_drn_layer_dims,
    resolve_logical_hidden_dim,
)
from .metrics import drn_saturation_fraction, q_abs_error
from .teacher import TeacherLayerActivations, apply_next_layer_norm


@dataclass
class SingleBlockLoss:
    loss: torch.Tensor
    metrics: dict[str, float | int | str]


class SingleBlockDRN(nn.Module):
    """Standalone tokenwise DRN trained to replace one frozen OPT MLP."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        drn_iter: int = 4,
        signed_drive: bool = True,
        drive_architecture: str = "projected_hidden",
        non_linearity: str = "perfect_diode",
        learn_drive_scale: bool = True,
        init_drive_scale: float = 1.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        learn_amplification: bool = False,
        weight_gains: float = 0.1,
        weight_min: float | None = 1.0e-5,
        weight_max: float | None = None,
        hard_sigmoid_param: dict[str, float] | None = None,
        bias_gain: float = 0.0,
        signed_output_weights: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.drive_architecture = str(drive_architecture)
        if self.drive_architecture == "signed_input_free":
            self.layer_dims = (2 * self.d_model, self.hidden_dim, self.d_model)
        else:
            if self.hidden_dim % 2 != 0:
                self.hidden_dim += 1
            self.layer_dims = (self.hidden_dim, self.d_model)
        self.register_buffer("input_scale", torch.tensor(float(max(input_scale, 1.0e-12))))
        self.register_buffer("output_scale", torch.tensor(float(max(output_scale, 1.0e-12))))
        self.output_gain = nn.Parameter(torch.tensor(1.0))
        self.drn = build_tokenwise_drn_mlp(
            d_model=self.d_model,
            layer_dims=self.layer_dims,
            drive_architecture=self.drive_architecture,
            signed_drive=bool(signed_drive),
            signed_output_weights=bool(signed_output_weights),
            dropout=dropout,
            drn_iter=int(drn_iter),
            non_linearity=str(non_linearity),
            weight_gains=float(weight_gains),
            bias_gain=float(bias_gain),
            voltage_amp=float(voltage_amp),
            current_amp=float(current_amp),
            learn_amplification=bool(learn_amplification),
            weight_min=weight_min,
            weight_max=weight_max,
            hard_sigmoid_param=hard_sigmoid_param,
            learn_drive_scale=bool(learn_drive_scale),
            init_drive_scale=float(init_drive_scale),
        )
        self.drn.enable_resistive_grad_()

    def forward(
        self,
        z: torch.Tensor,
        *,
        reset: bool | None = None,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        z_scaled = z / self.input_scale.clamp_min(1.0e-12)
        return self.output_gain * self.output_scale * self.drn(z_scaled, reset=reset, num_iterations=num_iterations)

    def optimizer_tensors(self) -> list[torch.Tensor]:
        return list(self.parameters()) + self.resistive_param_states()

    def resistive_param_states(self) -> list[torch.Tensor]:
        return self.drn.resistive_param_states()

    def named_resistive_parameters(self):
        yield from self.drn.named_resistive_parameters()

    def enable_resistive_grad_(self, enabled: bool = True) -> "SingleBlockDRN":
        self.drn.enable_resistive_grad_(enabled)
        return self

    def train_current_frontend_(self, enabled: bool = True) -> "SingleBlockDRN":
        for param in self.drn.block.ff.parameters():
            param.requires_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True) -> "SingleBlockDRN":
        self.drn.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self) -> "SingleBlockDRN":
        self.drn.clamp_resistive_params_()
        return self

    def detach_state_(self) -> "SingleBlockDRN":
        self.drn.detach_state_()
        return self

    def collect_diagnostics(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.drn.collect_diagnostics().items()}


def build_single_block_drn(
    teacher_layer: nn.Module,
    *,
    input_scale: float,
    output_scale: float,
    drn_iter: int,
    signed_drive: bool,
    drive_architecture: str = "projected_hidden",
    hidden_multiplier: float | None = None,
    weight_gains: float,
    bias_gain: float,
    init_drive_scale: float,
    voltage_amp: float = 1.0,
    current_amp: float = 1.0,
    learn_amplification: bool = False,
    init_mode: str = "random",
    learn_drive_scale: bool = True,
    weight_min: float | None = 1.0e-5,
    weight_max: float | None = None,
    non_linearity: str = "perfect_diode",
    hard_sigmoid_param: dict[str, float] | None = None,
    signed_output_weights: bool = False,
) -> SingleBlockDRN:
    embed_dim = int(teacher_layer.embed_dim)
    teacher_hidden_dim = int(teacher_layer.fc1.out_features)
    logical_hidden_dim = resolve_logical_hidden_dim(
        embed_dim=embed_dim,
        teacher_hidden_dim=teacher_hidden_dim,
        hidden_multiplier=hidden_multiplier,
    )
    layer_dims = resolve_drn_layer_dims(
        embed_dim=embed_dim,
        logical_hidden_dim=logical_hidden_dim,
        signed_drive=signed_drive,
        drive_architecture=drive_architecture,
    )
    hidden_dim = layer_dims[1] if drive_architecture == "signed_input_free" else layer_dims[0]

    if init_mode == "teacher_frontend_no_scale":
        learn_drive_scale = False
    block = SingleBlockDRN(
        d_model=embed_dim,
        hidden_dim=hidden_dim,
        input_scale=input_scale,
        output_scale=output_scale,
        drn_iter=drn_iter,
        signed_drive=signed_drive,
        drive_architecture=drive_architecture,
        non_linearity=non_linearity,
        learn_drive_scale=learn_drive_scale,
        init_drive_scale=init_drive_scale,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        learn_amplification=learn_amplification,
        weight_gains=weight_gains,
        weight_min=weight_min,
        weight_max=weight_max,
        hard_sigmoid_param=hard_sigmoid_param,
        bias_gain=bias_gain,
        signed_output_weights=signed_output_weights,
        dropout=0.0,
    )
    if init_mode in {"teacher_frontend", "teacher_frontend_no_scale", "teacher_frontend_scale"}:
        initialize_frontend_from_teacher_fc1(block, teacher_layer)
    elif init_mode != "random":
        raise ValueError(f"Unsupported init_mode '{init_mode}'.")
    return block


@torch.no_grad()
def initialize_frontend_from_teacher_fc1(block: SingleBlockDRN, teacher_layer: nn.Module) -> None:
    """Initialize the DRN current frontend from OPT fc1.

    This is intentionally scoped to the digital current frontend. A full
    conductance-level fc2/sign-split initialization is a separate ablation.
    """

    if block.drive_architecture != "projected_hidden":
        raise ValueError("teacher_frontend initialization is only defined for projected_hidden drive architecture.")
    signed_drive = bool(getattr(block.drn.block, "signed_drive", False))
    expected_frontend_dim = int(teacher_layer.fc1.out_features)
    if signed_drive:
        expected_physical_dim = 2 * expected_frontend_dim
        if block.hidden_dim != expected_physical_dim:
            raise ValueError(
                "signed teacher_frontend initialization requires a physical DRN hidden_dim "
                f"of 2 * teacher fc1.out_features ({expected_physical_dim}), got {block.hidden_dim}."
            )
    elif block.hidden_dim != expected_frontend_dim:
        raise ValueError(
            "teacher_frontend initialization requires the DRN hidden_dim to match teacher fc1.out_features."
        )
    linear = _find_first_linear(block.drn.block.ff)
    if linear.out_features != expected_frontend_dim:
        raise ValueError(
            "teacher_frontend initialization requires the current frontend output dimension "
            f"to match teacher fc1.out_features ({expected_frontend_dim}), got {linear.out_features}."
        )
    linear.weight.copy_(teacher_layer.fc1.weight.detach() * block.input_scale.detach())
    if linear.bias is None:
        if teacher_layer.fc1.bias is not None:
            raise ValueError("Teacher fc1 has a bias but the DRN frontend was built without one.")
    elif teacher_layer.fc1.bias is None:
        linear.bias.zero_()
    else:
        linear.bias.copy_(teacher_layer.fc1.bias.detach())


def single_block_loss(
    model: SingleBlockDRN,
    teacher: nn.Module,
    layer_index: int,
    activations: TeacherLayerActivations,
    *,
    objective: str,
    alpha_next_ln: float = 0.1,
    alpha_cosine: float = 0.1,
    alpha_norm: float = 0.1,
    alpha_post_residual: float = 0.0,
    alpha_logit_kl: float = 0.0,
    logit_temperature: float = 1.0,
) -> SingleBlockLoss:
    pred_r = model(activations.z, reset=True)
    local_mse = F.mse_loss(pred_r, activations.r)
    student_post = activations.a + pred_r
    post_residual_mse = F.mse_loss(student_post, activations.h_next)
    target_energy = torch.mean(activations.r.detach().float() ** 2).clamp_min(1.0e-12)
    cosine_loss = _cosine_loss(pred_r, activations.r)
    norm_ratio, norm_ratio_loss = _norm_ratio_loss(pred_r, activations.r)
    next_ln_mse = None
    final_logit_kl = None

    if objective == "local_mlp":
        loss = local_mse
    elif objective == "local_mlp_cosine":
        scale = target_energy.to(local_mse.device)
        loss = local_mse
        loss = loss + float(alpha_cosine) * scale * cosine_loss
        loss = loss + float(alpha_norm) * scale * norm_ratio_loss
    elif objective == "post_residual":
        loss = post_residual_mse
    elif objective == "next_ln_aux":
        next_ln_mse = _next_ln_mse(teacher, layer_index, activations, student_post)
        loss = post_residual_mse + float(alpha_next_ln) * next_ln_mse
    elif objective == "rigorous_pretrain":
        next_ln_mse = _next_ln_mse(teacher, layer_index, activations, student_post)
        loss = local_mse
        loss = loss + float(alpha_cosine) * target_energy.to(local_mse.device) * cosine_loss
        loss = loss + float(alpha_norm) * target_energy.to(local_mse.device) * norm_ratio_loss
        if alpha_post_residual > 0.0:
            loss = loss + float(alpha_post_residual) * post_residual_mse
        if alpha_next_ln > 0.0:
            loss = loss + float(alpha_next_ln) * next_ln_mse
        if alpha_logit_kl > 0.0:
            final_logit_kl = _final_layer_logit_kl(
                teacher,
                layer_index,
                student_post,
                activations.h_next.detach(),
                temperature=float(logit_temperature),
            )
            loss = loss + float(alpha_logit_kl) * final_logit_kl
    else:
        raise ValueError(f"Unsupported objective '{objective}'.")

    post_target_energy = torch.mean(activations.h_next.detach().float() ** 2).clamp_min(1.0e-12)
    metrics: dict[str, float | int | str] = {
        "objective": objective,
        "loss": float(loss.detach().item()),
        "mse": float(local_mse.detach().item()),
        "rel_mse": float((local_mse.detach() / target_energy).item()),
        "cosine": _cosine(pred_r.detach(), activations.r.detach()),
        "cosine_loss": float(cosine_loss.detach().item()),
        "norm_ratio": float(norm_ratio.detach().item()),
        "norm_ratio_loss": float(norm_ratio_loss.detach().item()),
        "post_residual_mse": float(post_residual_mse.detach().item()),
        "post_residual_rel_mse": float((post_residual_mse.detach() / post_target_energy).item()),
        "q99_abs_error": q_abs_error(pred_r.detach(), activations.r.detach(), q=0.99),
        "saturation_fraction": drn_saturation_fraction(model),
        "input_scale": float(model.input_scale.detach().item()),
        "output_scale": float(model.output_scale.detach().item()),
        "output_gain": float(model.output_gain.detach().item()),
        "drive_scale": float(model.drn.block.drive_scale.detach().item()),
    }
    if next_ln_mse is not None:
        metrics["next_ln_mse"] = float(next_ln_mse.detach().item())
    if final_logit_kl is not None:
        metrics["final_logit_kl"] = float(final_logit_kl.detach().item())
    try:
        for key, value in model.collect_diagnostics().items():
            metrics[f"drn/{key}"] = value
    except RuntimeError:
        pass
    return SingleBlockLoss(loss=loss, metrics=metrics)


def trainable_tensors(model: SingleBlockDRN) -> list[torch.Tensor]:
    return [tensor for tensor in _dedupe_tensors(model.optimizer_tensors()) if tensor.requires_grad]


def optimizer_param_groups(
    model: SingleBlockDRN,
    *,
    amp_lr: float | None = None,
    output_gain_lr: float | None = None,
) -> list[dict[str, Any]]:
    if amp_lr is not None:
        model.drn.block.amp_learning_rate = float(amp_lr)
    groups = []
    seen = set()
    for group in model.drn.optimizer_param_groups():
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
        groups.append(normalized)
    if model.output_gain.requires_grad and id(model.output_gain) not in seen:
        seen.add(id(model.output_gain))
        group: dict[str, Any] = {"params": [model.output_gain], "name": "output_gain"}
        gain_lr = output_gain_lr if output_gain_lr is not None else amp_lr
        if gain_lr is not None:
            group["lr"] = float(gain_lr)
            group["weight_decay"] = 0.0
        groups.append(group)
    if groups:
        return groups
    return [{"params": trainable_tensors(model)}]


def all_tensors(model: SingleBlockDRN) -> list[torch.Tensor]:
    return _dedupe_tensors(model.optimizer_tensors())


def _find_first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise RuntimeError("Could not find a Linear layer in the DRN frontend.")


def _next_ln_mse(
    teacher: nn.Module,
    layer_index: int,
    activations: TeacherLayerActivations,
    student_post: torch.Tensor,
) -> torch.Tensor:
    teacher_next = (
        activations.next_ln.detach()
        if activations.next_ln is not None
        else apply_next_layer_norm(teacher, layer_index, activations.h_next.detach())
    )
    student_next = apply_next_layer_norm(teacher, layer_index, student_post)
    return F.mse_loss(student_next, teacher_next)


def _final_layer_logit_kl(
    teacher: nn.Module,
    layer_index: int,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    layers = teacher.model.decoder.layers
    if int(layer_index) != len(layers) - 1:
        raise ValueError("final logit auxiliary is only valid for the final decoder layer.")
    if temperature <= 0.0:
        raise ValueError("logit_temperature must be positive.")
    with torch.no_grad():
        teacher_logits = _final_logits_from_hidden(teacher, teacher_hidden.detach()).float()
        teacher_probs = F.softmax(teacher_logits / float(temperature), dim=-1)
    student_logits = _final_logits_from_hidden(teacher, student_hidden).float()
    student_log_probs = F.log_softmax(student_logits / float(temperature), dim=-1)
    kl_per_token = torch.sum(
        teacher_probs * (torch.log(teacher_probs.clamp_min(1.0e-8)) - student_log_probs),
        dim=-1,
    )
    return kl_per_token.mean() * float(temperature) ** 2


def _final_logits_from_hidden(teacher: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    decoder = teacher.model.decoder
    final_layer_norm = getattr(decoder, "final_layer_norm", None)
    if final_layer_norm is not None:
        hidden = final_layer_norm(hidden)
    project_out = getattr(decoder, "project_out", None)
    if project_out is not None:
        hidden = project_out(hidden)
    return teacher.lm_head(hidden)


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


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    denom = left_norm * right_norm
    if float(denom.item()) <= 1.0e-12:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).div(denom).item())


def _cosine_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().reshape(1, -1)
    right_flat = right.detach().float().reshape(1, -1)
    cosine = F.cosine_similarity(left_flat, right_flat, dim=1, eps=1.0e-12)
    return 1.0 - cosine.mean()


def _norm_ratio_loss(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left.float().reshape(-1))
    right_norm = torch.linalg.vector_norm(right.detach().float().reshape(-1)).clamp_min(1.0e-12)
    ratio = left_norm / right_norm
    return ratio, (ratio - 1.0).pow(2)
