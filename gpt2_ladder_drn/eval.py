from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .utils import perplexity


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        out = model(x, targets=y)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("Model did not return a loss during evaluation.")
        losses.append(float(loss.item()))
    if was_training:
        model.train()

    mean_loss = sum(losses) / max(1, len(losses))
    return {"loss": mean_loss, "ppl": perplexity(mean_loss)}
