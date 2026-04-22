from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..blocks.base import BlockFreeCache, DigitalDRNBlock
from ..core.parameter import Bias
from ..models.network_digital_analog import DigitalAnalogNet
from .ep_block import BlockEPResult, BlockEquilibriumProp
from .ep_digital import DigitalVJPResult, vjp_ff_block


@dataclass
class BlockBackwardResult:
    ep: BlockEPResult
    digital: DigitalVJPResult


@dataclass
class NetworkFreeCache:
    block_caches: list[BlockFreeCache]
    features: torch.Tensor
    logits: torch.Tensor


@dataclass
class HeadBackwardResult:
    head_params: list[nn.Parameter]
    head_param_grads: list[torch.Tensor]
    delta_h: torch.Tensor


@dataclass
class HybridBackwardResult:
    free_cache: NetworkFreeCache
    head: HeadBackwardResult
    blocks: list[BlockBackwardResult]


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


def forward_free_with_cache(
    model: DigitalAnalogNet,
    inputs: torch.Tensor,
    *,
    reset: bool = False,
    num_iterations: int | None = None,
) -> NetworkFreeCache:
    h = inputs
    block_caches: list[BlockFreeCache] = []
    for block in model.blocks:
        if not isinstance(block, DigitalDRNBlock):
            raise TypeError("forward_free_with_cache currently expects DRN runtime blocks only.")
        h_prev = h
        h = block(h_prev, reset=reset, num_iterations=num_iterations)
        block_caches.append(block.capture_free_cache(h_prev))

    features = h.detach().clone()
    logits = features if model.head is None else model.head(features)
    return NetworkFreeCache(
        block_caches=block_caches,
        features=features,
        logits=logits,
    )


def backward_head(
    model: DigitalAnalogNet,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> HeadBackwardResult:
    head_params = []
    if model.head is not None:
        head_params = [param for param in model.head.parameters() if param.requires_grad]
    feature_leaf = features.detach().clone().requires_grad_(True)
    bn_state = _capture_batchnorm_buffers(model.head) if model.head is not None else []
    try:
        logits = feature_leaf if model.head is None else model.head(feature_leaf)
        loss = criterion(logits, targets)
        grads = torch.autograd.grad(
            loss,
            (feature_leaf, *head_params),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    finally:
        _restore_batchnorm_buffers(bn_state)
    delta_h = grads[0].detach().clone()
    head_param_grads = [grad.detach().clone() for grad in grads[1:]]
    return HeadBackwardResult(
        head_params=head_params,
        head_param_grads=head_param_grads,
        delta_h=delta_h,
    )


def hybrid_backward_explicit(
    model: DigitalAnalogNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    criterion: nn.Module,
    beta: float,
    amp_gradient_compensation: bool = False,
    reset: bool = False,
    num_iterations: int | None = None,
) -> HybridBackwardResult:
    free_cache = forward_free_with_cache(
        model,
        inputs,
        reset=reset,
        num_iterations=num_iterations,
    )
    head = backward_head(model, free_cache.features, targets, criterion)

    block_results_reversed: list[BlockBackwardResult] = []
    delta_h = head.delta_h
    downstream_weight_count = 0
    for block, cache in zip(reversed(model.blocks), reversed(free_cache.block_caches)):
        if not isinstance(block, DigitalDRNBlock):
            raise TypeError("hybrid_backward_explicit currently expects DRN runtime blocks only.")
        weight_count = sum(0 if isinstance(param, Bias) else 1 for param in block.resistive_params())
        ep = BlockEquilibriumProp(
            block,
            beta=beta,
            amp_gradient_compensation=amp_gradient_compensation,
            downstream_weight_count=downstream_weight_count,
            num_iterations=num_iterations,
        ).compute_gradients(
            free_cache=cache,
            output_cotangent=delta_h,
        )
        digital = vjp_ff_block(block, cache.h_prev, ep.delta_drive)
        block_results_reversed.append(BlockBackwardResult(ep=ep, digital=digital))
        delta_h = digital.delta_h_prev
        downstream_weight_count += weight_count

    block_results_reversed.reverse()
    return HybridBackwardResult(
        free_cache=free_cache,
        head=head,
        blocks=block_results_reversed,
    )
