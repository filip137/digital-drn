from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from ..blocks.base import BlockFreeCache
from ..models.tokenwise import TokenwiseDRNMLP
from ..models.transformer import SmallDRNGPT
from .ep_block import BlockEPResult, BlockEquilibriumProp
from .ep_digital import DigitalVJPResult, vjp_ff_block


@dataclass
class TransformerDRNBlockEPDiagnostic:
    name: str
    mlp: TokenwiseDRNMLP
    free_cache: BlockFreeCache
    output_cotangent: torch.Tensor
    bp_ff_grads: list[torch.Tensor]
    bp_drive_grad: torch.Tensor | None
    bp_drn_grads: list[torch.Tensor]
    ep: BlockEPResult
    digital: DigitalVJPResult


@dataclass
class TransformerEPDiagnostic:
    loss: float
    logits: torch.Tensor
    bp_groups: OrderedDict[str, list[torch.Tensor]]
    ep_groups: OrderedDict[str, list[torch.Tensor]]
    group_metrics: list[dict[str, float | str]]
    overall_metrics: dict[str, float]
    displacement: dict[str, float]
    blocks: list[TransformerDRNBlockEPDiagnostic]


@dataclass
class _ForwardRecord:
    name: str
    mlp: TokenwiseDRNMLP
    free_cache: BlockFreeCache
    output_cotangent: torch.Tensor | None = None


def _clone_grad_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        return torch.zeros_like(tensor.detach())
    return tensor.grad.detach().clone()


def _tensor_list_cpu(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().cpu().float().clone() for tensor in tensors]


def _flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flat = [tensor.reshape(-1).float() for tensor in tensors]
    if not flat:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(flat)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    norm_a = float(a.norm().item())
    norm_b = float(b.norm().item())
    if norm_a < 1.0e-12 and norm_b < 1.0e-12:
        return 1.0
    if norm_a < 1.0e-12 or norm_b < 1.0e-12:
        return float("nan")
    return float(torch.dot(a, b).item() / (norm_a * norm_b))


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    norm_a = float(a.norm().item())
    if norm_a < 1.0e-12:
        return float("nan")
    return float((a - b).norm().item() / norm_a)


def _group_metrics(
    bp_groups: OrderedDict[str, list[torch.Tensor]],
    ep_groups: OrderedDict[str, list[torch.Tensor]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in bp_groups:
        bp_flat = _flatten(bp_groups[name])
        ep_flat = _flatten(ep_groups[name])
        rows.append(
            {
                "group": name,
                "cosine": _cosine(bp_flat, ep_flat),
                "relative_error": _relative_error(bp_flat, ep_flat),
                "bp_norm": float(bp_flat.norm().item()),
                "ep_norm": float(ep_flat.norm().item()),
            }
        )
    return rows


def _overall_metrics(
    bp_groups: OrderedDict[str, list[torch.Tensor]],
    ep_groups: OrderedDict[str, list[torch.Tensor]],
) -> dict[str, float]:
    bp_flat = _flatten(tensor for tensors in bp_groups.values() for tensor in tensors)
    ep_flat = _flatten(tensor for tensors in ep_groups.values() for tensor in tensors)
    return {
        "overall_cosine": _cosine(bp_flat, ep_flat),
        "overall_relative_error": _relative_error(bp_flat, ep_flat),
        "bp_norm": float(bp_flat.norm().item()),
        "ep_norm": float(ep_flat.norm().item()),
    }


def _pool_displacement(results: list[TransformerDRNBlockEPDiagnostic]) -> dict[str, float]:
    def _accumulate(perturbed_name: str) -> tuple[float, float]:
        total_free_sq = 0.0
        total_diff_sq = 0.0
        for result in results:
            perturbed_layers = getattr(result.ep, perturbed_name)
            for free, perturbed in zip(result.free_cache.free_state, perturbed_layers):
                free = free.detach().float()
                perturbed = perturbed.detach().float()
                diff = perturbed - free
                total_free_sq += float(torch.sum(free * free).item())
                total_diff_sq += float(torch.sum(diff * diff).item())
        return total_free_sq, total_diff_sq

    pos_free_sq, pos_diff_sq = _accumulate("plus_state")
    neg_free_sq, neg_diff_sq = _accumulate("minus_state")

    def _relative(diff_sq: float, free_sq: float) -> float:
        return math.sqrt(diff_sq / free_sq) if free_sq > 1.0e-12 else float("nan")

    positive = _relative(pos_diff_sq, pos_free_sq)
    negative = _relative(neg_diff_sq, neg_free_sq)
    return {
        "positive_relative_disp": positive,
        "negative_relative_disp": negative,
        "mean_relative_disp": 0.5 * (positive + negative),
    }


def _sequence_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if (
        isinstance(criterion, nn.CrossEntropyLoss)
        and logits.ndim >= 3
        and targets.shape == logits.shape[:-1]
    ):
        return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return criterion(logits, targets)


def _collect_forward_records(model: SmallDRNGPT) -> tuple[list[_ForwardRecord], list[torch.utils.hooks.RemovableHandle]]:
    records: list[_ForwardRecord] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, TokenwiseDRNMLP):
            continue

        def _make_hook(name: str, mlp: TokenwiseDRNMLP):
            def _hook(_block, args, output):
                h_prev = args[0]
                record = _ForwardRecord(
                    name=name,
                    mlp=mlp,
                    free_cache=mlp.block.capture_free_cache(h_prev),
                )

                def _capture_cotangent(grad: torch.Tensor) -> torch.Tensor:
                    record.output_cotangent = grad.detach().clone()
                    return grad

                output.register_hook(_capture_cotangent)
                records.append(record)

            return _hook

        handles.append(module.block.register_forward_hook(_make_hook(module_name, module)))

    return records, handles


def transformer_drn_ep_diagnostic(
    model: SmallDRNGPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    criterion: nn.Module,
    beta: float,
    reset: bool = True,
    num_iterations: int | None = None,
    amp_gradient_compensation: bool = False,
    eval_mode: bool = True,
) -> TransformerEPDiagnostic:
    """Compare BP and local EP gradients for tokenwise DRN MLPs in a GPT model.

    This is diagnostic plumbing only. It intentionally does not assign EP
    gradients to the full transformer or step an optimizer.
    """

    if beta <= 0.0:
        raise ValueError("beta must be strictly positive for transformer EP diagnostics.")

    was_training = model.training
    if eval_mode:
        model.eval()

    def _restore_model_state() -> None:
        model.detach_state_()
        if eval_mode and was_training:
            model.train()

    model.zero_grad(set_to_none=True)
    model.zero_resistive_grad_(set_to_none=True)

    records, handles = _collect_forward_records(model)
    try:
        logits = model(input_ids, reset=reset, num_iterations=num_iterations)
        loss = _sequence_loss(criterion, logits, targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    if not records:
        _restore_model_state()
        raise ValueError("No TokenwiseDRNMLP modules were observed during the transformer forward pass.")

    results: list[TransformerDRNBlockEPDiagnostic] = []
    bp_groups: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
    ep_groups: OrderedDict[str, list[torch.Tensor]] = OrderedDict()

    for record in records:
        if record.output_cotangent is None:
            _restore_model_state()
            raise RuntimeError(f"Missing output cotangent for transformer DRN MLP '{record.name}'.")

        block = record.mlp.block
        ff_params = [param for param in block.ff.parameters() if param.requires_grad]
        bp_ff_grads = [_clone_grad_or_zero(param) for param in ff_params]
        bp_drive_grad = None
        if block._drive_scale_raw.requires_grad:
            bp_drive_grad = _clone_grad_or_zero(block._drive_scale_raw)
        bp_drn_grads = [_clone_grad_or_zero(param.state) for param in block.resistive_params()]

        ep = BlockEquilibriumProp(
            block,
            beta=beta,
            amp_gradient_compensation=amp_gradient_compensation,
            num_iterations=num_iterations,
        ).compute_gradients(
            free_cache=record.free_cache,
            output_cotangent=record.output_cotangent,
        )
        digital = vjp_ff_block(block, record.free_cache.h_prev, ep.delta_drive)

        result = TransformerDRNBlockEPDiagnostic(
            name=record.name,
            mlp=record.mlp,
            free_cache=record.free_cache,
            output_cotangent=record.output_cotangent.detach().clone(),
            bp_ff_grads=bp_ff_grads,
            bp_drive_grad=bp_drive_grad,
            bp_drn_grads=bp_drn_grads,
            ep=ep,
            digital=digital,
        )
        results.append(result)

        bp_groups[f"{record.name}/ff"] = _tensor_list_cpu(bp_ff_grads)
        ep_groups[f"{record.name}/ff"] = _tensor_list_cpu(digital.ff_param_grads)

        bp_groups[f"{record.name}/drive"] = _tensor_list_cpu([] if bp_drive_grad is None else [bp_drive_grad])
        ep_groups[f"{record.name}/drive"] = _tensor_list_cpu(
            [] if digital.drive_scale_grad is None else [digital.drive_scale_grad]
        )

        bp_groups[f"{record.name}/drn"] = _tensor_list_cpu(bp_drn_grads)
        ep_groups[f"{record.name}/drn"] = _tensor_list_cpu(ep.param_grads)

    try:
        return TransformerEPDiagnostic(
            loss=float(loss.detach().item()),
            logits=logits.detach().clone(),
            bp_groups=bp_groups,
            ep_groups=ep_groups,
            group_metrics=_group_metrics(bp_groups, ep_groups),
            overall_metrics=_overall_metrics(bp_groups, ep_groups),
            displacement=_pool_displacement(results),
            blocks=results,
        )
    finally:
        _restore_model_state()
