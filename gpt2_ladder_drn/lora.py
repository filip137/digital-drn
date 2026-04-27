from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be strictly positive.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(base.in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.B(self.A(self.dropout(x)))


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("c_attn", "c_proj"),
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    replaced = _replace_lora_modules(model, target_modules, r, alpha, dropout)
    if replaced == 0:
        targets = ", ".join(target_modules)
        raise ValueError(f"No Linear modules matched LoRA targets: {targets}.")
    return model


def _replace_lora_modules(
    module: nn.Module,
    target_modules: Iterable[str],
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    targets = set(target_modules)
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if child_name in targets and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += _replace_lora_modules(child, targets, r, alpha, dropout)
    return replaced
