from __future__ import annotations

import random
from typing import Any

import torch

try:  # pragma: no cover - numpy is expected but optional in minimal installs
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested CUDA device '{device}' but CUDA is not available.")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device index {resolved.index} but only {torch.cuda.device_count()} device(s) are available."
            )
        if resolved.index is None:
            resolved = torch.device("cuda:0")
    return resolved


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def tensor_stats(tensor: torch.Tensor, prefix: str, eps: float = 1.0e-12) -> dict[str, Any]:
    values = tensor.detach().float()
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.unsqueeze(0)
    else:
        values = values.flatten(start_dim=1)

    rms = values.square().mean().sqrt()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_rms": float(rms.item()),
        f"{prefix}_max_abs": float(values.abs().max().item()),
        f"{prefix}_near_zero_frac": float((values.abs() < eps).float().mean().item()),
    }
