from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..blocks.base import DigitalDRNBlock


@dataclass
class DigitalVJPResult:
    ff_params: list[nn.Parameter]
    ff_param_grads: list[torch.Tensor]
    drive_scale_grad: torch.Tensor | None
    delta_h_prev: torch.Tensor


def _capture_batchnorm_buffers(module: nn.Module) -> list[tuple[_BatchNorm, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    state = []
    for child in module.modules():
        if not isinstance(child, _BatchNorm):
            continue
        running_mean = None if child.running_mean is None else child.running_mean.detach().clone()
        running_var = None if child.running_var is None else child.running_var.detach().clone()
        num_batches = None if child.num_batches_tracked is None else child.num_batches_tracked.detach().clone()
        state.append((child, running_mean, running_var, num_batches))
    return state


def _restore_batchnorm_buffers(
    state: list[tuple[_BatchNorm, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]
) -> None:
    for child, running_mean, running_var, num_batches in state:
        if running_mean is not None and child.running_mean is not None:
            child.running_mean.copy_(running_mean)
        if running_var is not None and child.running_var is not None:
            child.running_var.copy_(running_var)
        if num_batches is not None and child.num_batches_tracked is not None:
            child.num_batches_tracked.copy_(num_batches)


def vjp_ff_block(
    block: DigitalDRNBlock,
    h_prev: torch.Tensor,
    delta_drive: torch.Tensor,
) -> DigitalVJPResult:
    """Backpropagate an EP drive cotangent through x = drive_scale * ff(h_prev)."""

    h_prev_leaf = h_prev.detach().clone().requires_grad_(True)
    ff_params = [param for param in block.ff.parameters() if param.requires_grad]

    inputs: list[torch.Tensor] = [h_prev_leaf, *ff_params]
    if block._drive_scale_raw.requires_grad:
        inputs.append(block._drive_scale_raw)

    bn_state = _capture_batchnorm_buffers(block.ff)
    try:
        drive = block.drive_scale * block.ff(h_prev_leaf)
        grads = torch.autograd.grad(
            outputs=drive,
            inputs=tuple(inputs),
            grad_outputs=delta_drive,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    finally:
        _restore_batchnorm_buffers(bn_state)

    delta_h_prev = grads[0].detach().clone()
    ff_param_grads = [grad.detach().clone() for grad in grads[1 : 1 + len(ff_params)]]
    drive_scale_grad = None
    if block._drive_scale_raw.requires_grad:
        drive_scale_grad = grads[-1].detach().clone()

    return DigitalVJPResult(
        ff_params=ff_params,
        ff_param_grads=ff_param_grads,
        drive_scale_grad=drive_scale_grad,
        delta_h_prev=delta_h_prev,
    )
