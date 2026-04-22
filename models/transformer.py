from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenwise import TokenwiseDRNMLP


@dataclass
class DRNGPTConfig:
    vocab_size: int = 512
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = False
    drn_hidden_dim: int | None = None
    drn_num_iterations: int = 6
    drn_mode: str = "asynchronous"
    drn_non_linearity: str = "perfect_diode"
    drn_ff_activation: str | None = "identity"
    drn_signed_drive: bool = False
    drn_weight_gains: float = 0.1
    drn_bias_gain: float = 0.0
    drn_voltage_amp: float = 1.0
    drn_current_amp: float = 1.0
    drn_learn_drive_scale: bool = True
    drn_init_drive_scale: float = 1.0
    drn_weight_min: float | None = None
    drn_weight_max: float | None = None
    drn_weight_init_mode: str = "kaiming_uniform"
    drn_reset_each_forward: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1.0e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DRNGPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = int(config.n_heads)
        self.head_dim = int(config.d_model // config.n_heads)
        self.dropout = float(config.dropout)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

        mask = torch.tril(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.seq_len, config.seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(~causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(y))


class DRNTransformerBlock(nn.Module):
    def __init__(self, config: DRNGPTConfig) -> None:
        super().__init__()
        hidden_dim = int(
            config.drn_hidden_dim if config.drn_hidden_dim is not None else config.mlp_ratio * config.d_model
        )
        if hidden_dim <= 0:
            raise ValueError("DRN hidden dimension must be strictly positive.")
        if hidden_dim % 2 != 0:
            raise ValueError("DRN hidden dimension must be even.")

        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = TokenwiseDRNMLP(
            d_model=config.d_model,
            hidden_dim=hidden_dim,
            dropout=config.dropout,
            reset_each_forward=config.drn_reset_each_forward,
            ff_activation=config.drn_ff_activation,
            signed_drive=config.drn_signed_drive,
            num_iterations=config.drn_num_iterations,
            mode=config.drn_mode,
            non_linearity=config.drn_non_linearity,
            weight_gains=config.drn_weight_gains,
            bias_gain=config.drn_bias_gain,
            voltage_amp=config.drn_voltage_amp,
            current_amp=config.drn_current_amp,
            weight_min=config.drn_weight_min,
            weight_max=config.drn_weight_max,
            weight_init_mode=config.drn_weight_init_mode,
            learn_drive_scale=config.drn_learn_drive_scale,
            init_drive_scale=config.drn_init_drive_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        reset: bool = True,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), reset=reset, num_iterations=num_iterations)
        return x


class SmallDRNGPT(nn.Module):
    """Decoder-only GPT-style model with tokenwise DRN MLP sublayers."""

    def __init__(self, config: DRNGPTConfig) -> None:
        super().__init__()
        if config.seq_len <= 0:
            raise ValueError("seq_len must be strictly positive.")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be strictly positive.")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be strictly positive.")

        self.config = config
        self.tok_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DRNTransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            self.lm_head.weight = self.tok_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def drn_mlps(self) -> list[TokenwiseDRNMLP]:
        return [block.mlp for block in self.blocks]

    def set_device(self, device: torch.device | str):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        super().to(device)
        for mlp in self.drn_mlps():
            mlp.set_device(device)
        return self

    def resistive_params(self):
        return [param for mlp in self.drn_mlps() for param in mlp.resistive_params()]

    def resistive_param_states(self):
        return [param.state for param in self.resistive_params()]

    def optimizer_tensors(self):
        return list(self.parameters()) + self.resistive_param_states()

    def optimizer_param_groups(self):
        groups = [{"params": list(self.parameters())}]
        resistive_states = self.resistive_param_states()
        if resistive_states:
            groups.append({"params": resistive_states})
        return groups

    def enable_resistive_grad_(self, enabled: bool = True):
        for mlp in self.drn_mlps():
            mlp.enable_resistive_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True):
        for mlp in self.drn_mlps():
            mlp.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self):
        for mlp in self.drn_mlps():
            mlp.clamp_resistive_params_()
        return self

    def detach_state_(self):
        for mlp in self.drn_mlps():
            mlp.detach_state_()
        return self

    def named_resistive_parameters(self):
        for block_idx, mlp in enumerate(self.drn_mlps()):
            for name, tensor in mlp.named_resistive_parameters():
                yield f"blocks.{block_idx}.mlp.{name}", tensor

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        reset: bool = True,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        _batch_size, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"input sequence length {seq_len} exceeds model seq_len {self.config.seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.tok_embedding(input_ids) + self.pos_embedding(positions).unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, reset=reset, num_iterations=num_iterations)
        x = self.final_ln(x)
        return self.lm_head(x)

    def num_parameters(self, *, include_resistive: bool = True) -> int:
        total = sum(param.numel() for param in self.parameters())
        if include_resistive:
            total += sum(param.numel() for param in self.resistive_param_states())
        return total
