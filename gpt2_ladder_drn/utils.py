from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cycle(iterable: Iterable) -> Iterator:
    while True:
        for item in iterable:
            yield item


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 20.0 else float("inf")


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def cuda_peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)
