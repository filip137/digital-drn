from __future__ import annotations

import math

import torch
import torch.nn as nn

from digital_drn.models.tokenwise import TokenwiseDRNMLP

from .config import GPT2Config
from .model_gpt2 import CausalSelfAttention, LayerNorm


class DRNCell(nn.Module):
    """Small standalone fixed-point proxy retained for toy ablations.

    The DRN-LST side blocks below use the repository's real TokenwiseDRNMLP,
    which drives a DigitalDRNBlock and QuadraticMinimizer.
    """

    def __init__(
        self,
        d_model: int,
        n_iter: int = 8,
        damping: float = 0.5,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be strictly positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be strictly positive.")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1].")

        self.d_model = int(d_model)
        self.n_iter = int(n_iter)
        self.damping = float(damping)
        self.in_proj = nn.Linear(d_model, d_model)
        self.W_raw = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.W_raw, mean=0.0, std=init_scale)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)
        z = torch.zeros_like(u)
        w_sym = 0.5 * (self.W_raw + self.W_raw.t())
        w_sym = w_sym / math.sqrt(self.d_model)

        for _ in range(self.n_iter):
            pre = u + torch.matmul(z, w_sym.t()) + self.bias
            z_new = torch.tanh(pre)
            z = (1.0 - self.damping) * z + self.damping * z_new

        return self.out_proj(z)


class SideDRNBlock(nn.Module):
    def __init__(
        self,
        side_config: GPT2Config,
        drn_iter: int = 8,
        drn_damping: float = 0.5,
        signed_drive: bool = True,
        hidden_multiplier: int = 4,
    ) -> None:
        super().__init__()
        del drn_damping  # Coordinate descent has no damping knob; kept for CLI compatibility.
        self.ln_1 = LayerNorm(side_config.n_embd, bias=side_config.bias)
        self.attn = CausalSelfAttention(side_config)
        self.ln_2 = LayerNorm(side_config.n_embd, bias=side_config.bias)
        self.drn = _build_tokenwise_drn(
            side_config,
            drn_iter=drn_iter,
            dropout=side_config.dropout,
            signed_drive=signed_drive,
            hidden_multiplier=hidden_multiplier,
        )
        self.drn.enable_resistive_grad_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        # x already contains the gated ladder projection of frozen backbone
        # activations; TokenwiseDRNMLP turns it into injected DRN current.
        x = x + self.drn(self.ln_2(x), reset=True)
        return x

    def forward_post_drn_residual(
        self,
        q: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q + self.attn(self.ln_1(q))
        d = self.drn(self.ln_2(q), reset=True)
        return q + alpha * d, q, d


class PureDRNSideBlock(nn.Module):
    def __init__(
        self,
        side_config: GPT2Config,
        drn_iter: int = 8,
        drn_damping: float = 0.5,
        signed_drive: bool = True,
        hidden_multiplier: int = 4,
    ) -> None:
        super().__init__()
        del drn_damping  # Coordinate descent has no damping knob; kept for CLI compatibility.
        self.ln = LayerNorm(side_config.n_embd, bias=side_config.bias)
        self.drn = _build_tokenwise_drn(
            side_config,
            drn_iter=drn_iter,
            dropout=side_config.dropout,
            signed_drive=signed_drive,
            hidden_multiplier=hidden_multiplier,
        )
        self.drn.enable_resistive_grad_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The pure ablation still receives the ladder-mixed backbone signal.
        return x + self.drn(self.ln(x), reset=True)

    def forward_post_drn_residual(
        self,
        q: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.drn(self.ln(q), reset=True)
        return q + alpha * d, q, d


def _build_tokenwise_drn(
    side_config: GPT2Config,
    drn_iter: int,
    dropout: float,
    signed_drive: bool = True,
    hidden_multiplier: int = 4,
) -> TokenwiseDRNMLP:
    if hidden_multiplier <= 0:
        raise ValueError("hidden_multiplier must be strictly positive.")
    hidden_dim = int(hidden_multiplier) * side_config.n_embd
    if hidden_dim % 2 != 0:
        hidden_dim += 1
    return TokenwiseDRNMLP(
        d_model=side_config.n_embd,
        hidden_dim=hidden_dim,
        dropout=dropout,
        reset_each_forward=True,
        ff_activation="identity",
        signed_drive=signed_drive,
        num_iterations=drn_iter,
        mode="asynchronous",
        non_linearity="perfect_diode",
        weight_gains=0.1,
        bias_gain=0.0,
        voltage_amp=1.0,
        current_amp=1.0,
        weight_init_mode="kaiming_uniform",
        learn_drive_scale=True,
        init_drive_scale=1.0,
    )
