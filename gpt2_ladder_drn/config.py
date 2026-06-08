from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


@dataclass
class DebugGPT2Config:
    vocab_size: int = 512
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
