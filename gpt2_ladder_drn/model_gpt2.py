from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPT2Config


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1.0e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_dim = int(config.n_embd // config.n_head)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            att = att.masked_fill(~causal_mask, torch.finfo(att.dtype).min)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if config.block_size <= 0:
            raise ValueError("block_size must be strictly positive.")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be strictly positive.")
        if config.n_layer <= 0:
            raise ValueError("n_layer must be strictly positive.")

        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        batch_size, seq_len = input_ids.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {seq_len}; block size is {self.config.block_size}.")

        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        hidden_states: list[torch.Tensor] | None = [x] if return_hidden_states else None
        for block in self.transformer.h:
            x = block(x)
            if hidden_states is not None:
                hidden_states.append(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return {"logits": logits, "loss": loss, "hidden_states": hidden_states}

    @classmethod
    def from_hf_gpt2(cls, model_name: str = "gpt2") -> "GPT2LMHeadModel":
        try:
            from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel
        except ImportError as exc:
            raise ImportError("Install transformers to load Hugging Face GPT-2 checkpoints.") from exc

        hf_model = HFGPT2LMHeadModel.from_pretrained(model_name)
        hf_cfg = hf_model.config
        config = GPT2Config(
            vocab_size=hf_cfg.vocab_size,
            block_size=hf_cfg.n_positions,
            n_layer=hf_cfg.n_layer,
            n_head=hf_cfg.n_head,
            n_embd=hf_cfg.n_embd,
            dropout=hf_cfg.resid_pdrop,
            bias=True,
        )
        model = cls(config)

        with torch.no_grad():
            model.transformer.wte.weight.copy_(hf_model.transformer.wte.weight)
            model.transformer.wpe.weight.copy_(hf_model.transformer.wpe.weight)
            model.transformer.ln_f.weight.copy_(hf_model.transformer.ln_f.weight)
            model.transformer.ln_f.bias.copy_(hf_model.transformer.ln_f.bias)
            for ours, theirs in zip(model.transformer.h, hf_model.transformer.h):
                ours.ln_1.weight.copy_(theirs.ln_1.weight)
                ours.ln_1.bias.copy_(theirs.ln_1.bias)
                ours.ln_2.weight.copy_(theirs.ln_2.weight)
                ours.ln_2.bias.copy_(theirs.ln_2.bias)
                _copy_hf_conv1d(theirs.attn.c_attn, ours.attn.c_attn)
                _copy_hf_conv1d(theirs.attn.c_proj, ours.attn.c_proj)
                _copy_hf_conv1d(theirs.mlp.c_fc, ours.mlp.c_fc)
                _copy_hf_conv1d(theirs.mlp.c_proj, ours.mlp.c_proj)

        model.lm_head.weight = model.transformer.wte.weight
        return model


def _copy_hf_conv1d(src: nn.Module, dst: nn.Linear) -> None:
    dst.weight.copy_(src.weight.t())
    if dst.bias is not None and src.bias is not None:
        dst.bias.copy_(src.bias)
