from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DigitalGPTConfig:
    """Configuration for the fully-digital MQAR baseline transformer."""

    vocab_size: int = 512
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1.0e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DigitalGPTConfig) -> None:
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


class MLP(nn.Module):
    def __init__(self, config: DigitalGPTConfig) -> None:
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.d_model)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: DigitalGPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SmallDigitalGPT(nn.Module):
    """Fully-digital, causal, decoder-only transformer used for MQAR baselines."""

    def __init__(self, config: DigitalGPTConfig) -> None:
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
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
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

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"input sequence length {seq_len} exceeds model seq_len {self.config.seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.tok_embedding(input_ids) + self.pos_embedding(positions).unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
