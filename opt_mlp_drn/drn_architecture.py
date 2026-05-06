from __future__ import annotations

import torch
import torch.nn as nn

from digital_drn.blocks.block import build_dense_drn_block
from digital_drn.models.tokenwise import TokenwiseDRNMLP


DRN_DRIVE_ARCHITECTURES = ("projected_hidden", "signed_input_free")


class SignedIdentityDriveFrontend(nn.Module):
    """Inject the normalized transformer state directly as [z, -z]."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("SignedIdentityDriveFrontend expects a batched tensor.")
        x = x.reshape(x.size(0), -1)
        if x.size(1) != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {x.size(1)}.")
        return torch.cat((x, -x), dim=1)

    def optimizer_param_groups(self):
        return []


def resolve_logical_hidden_dim(
    *,
    embed_dim: int,
    teacher_hidden_dim: int,
    hidden_multiplier: float | None,
) -> int:
    logical_hidden_dim = int(teacher_hidden_dim)
    if hidden_multiplier is not None:
        logical_hidden_dim = int(round(float(hidden_multiplier) * int(embed_dim)))
    if logical_hidden_dim <= 0:
        raise ValueError("DRN hidden dimension must be strictly positive.")
    return logical_hidden_dim


def resolve_drn_layer_dims(
    *,
    embed_dim: int,
    logical_hidden_dim: int,
    signed_drive: bool,
    drive_architecture: str,
) -> tuple[int, ...]:
    if drive_architecture not in DRN_DRIVE_ARCHITECTURES:
        raise ValueError(
            f"Unsupported DRN drive architecture '{drive_architecture}'. "
            f"Expected one of {DRN_DRIVE_ARCHITECTURES}."
        )
    if drive_architecture == "projected_hidden":
        first_dim = int(logical_hidden_dim) * (2 if signed_drive else 1)
        return (first_dim, int(embed_dim))
    if not signed_drive:
        raise ValueError("signed_input_free requires signed_drive=True.")
    return (2 * int(embed_dim), 2 * int(logical_hidden_dim), int(embed_dim))


def build_tokenwise_drn_mlp(
    *,
    d_model: int,
    layer_dims: tuple[int, ...],
    drive_architecture: str,
    signed_drive: bool,
    signed_output_weights: bool,
    dropout: float,
    drn_iter: int,
    non_linearity: str,
    weight_gains: float,
    bias_gain: float,
    voltage_amp: float,
    current_amp: float,
    learn_amplification: bool,
    weight_min: float | None,
    weight_max: float | None,
    hard_sigmoid_param: dict[str, float] | None,
    learn_drive_scale: bool,
    init_drive_scale: float,
) -> TokenwiseDRNMLP:
    custom_block_required = drive_architecture == "signed_input_free" or bool(signed_output_weights)
    drn_block = None
    if custom_block_required:
        signed_edges = (len(layer_dims) - 2,) if signed_output_weights else None
        frontend = SignedIdentityDriveFrontend(d_model) if drive_architecture == "signed_input_free" else None
        block_kwargs = {
            "input_dim": int(d_model),
            "layer_dims": layer_dims,
            "ff": frontend,
            "ff_bias": True,
            "ff_activation": "identity",
            "signed_drive": bool(signed_drive) if frontend is None else False,
            "num_iterations": int(drn_iter),
            "mode": "asynchronous",
            "non_linearity": str(non_linearity),
            "weight_gains": float(weight_gains),
            "bias_gain": float(bias_gain),
            "voltage_amp": float(voltage_amp),
            "current_amp": float(current_amp),
            "learn_voltage_amp": bool(learn_amplification),
            "learn_current_amp": bool(learn_amplification),
            "weight_min": weight_min,
            "weight_max": weight_max,
            "hard_sigmoid_param": hard_sigmoid_param,
            "weight_init_mode": "kaiming_uniform",
            "learn_drive_scale": bool(learn_drive_scale),
            "init_drive_scale": float(init_drive_scale),
        }
        if signed_edges is not None:
            block_kwargs["signed_weight_edges"] = signed_edges
        drn_block = build_dense_drn_block(**block_kwargs)
        drn_block.drive_architecture = drive_architecture
        drn_block.signed_input_free = drive_architecture == "signed_input_free"

    drn_kwargs = {"block": drn_block} if drn_block is not None else {"hidden_dim": int(layer_dims[0])}
    module = TokenwiseDRNMLP(
        d_model=int(d_model),
        **drn_kwargs,
        dropout=float(dropout),
        reset_each_forward=True,
        ff_activation="identity",
        signed_drive=bool(signed_drive),
        num_iterations=int(drn_iter),
        mode="asynchronous",
        non_linearity=str(non_linearity),
        weight_gains=float(weight_gains),
        bias_gain=float(bias_gain),
        voltage_amp=float(voltage_amp),
        current_amp=float(current_amp),
        learn_voltage_amp=bool(learn_amplification),
        learn_current_amp=bool(learn_amplification),
        weight_min=weight_min,
        weight_max=weight_max,
        hard_sigmoid_param=hard_sigmoid_param,
        weight_init_mode="kaiming_uniform",
        learn_drive_scale=bool(learn_drive_scale),
        init_drive_scale=float(init_drive_scale),
    )
    module.block.drive_architecture = drive_architecture
    module.block.layer_dims = tuple(layer_dims)
    return module
