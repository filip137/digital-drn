from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def load_single_block_checkpoint_into_layer(checkpoint_path: str | Path, wrapper_layer) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    drn_state = {
        key.removeprefix("drn."): value
        for key, value in state.items()
        if key.startswith("drn.")
    }
    if not drn_state:
        drn_state = {
            key.removeprefix("drn_mlp."): value
            for key, value in state.items()
            if key.startswith("drn_mlp.")
        }
    if not drn_state:
        raise RuntimeError(f"{checkpoint_path} does not look like a single-block DRN checkpoint.")

    drn_state = _normalize_legacy_drn_state_keys(drn_state)
    wrapper_layer.drn_mlp.load_state_dict(drn_state, strict=True)
    resistive = checkpoint.get("resistive_parameters", {})
    if resistive:
        _load_resistive_parameters(checkpoint_path, resistive, wrapper_layer.drn_mlp)
    input_scale = state.get("input_scale", torch.tensor(1.0))
    output_scale = state.get("output_scale", torch.tensor(1.0))
    wrapper_layer.set_drn_scales(input_scale, output_scale)
    output_gain = state.get("output_gain", torch.tensor(1.0))
    if hasattr(wrapper_layer, "set_drn_output_gain"):
        wrapper_layer.set_drn_output_gain(output_gain)


def load_single_block_checkpoint_into_single_block(checkpoint_path: str | Path, block_model) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict) or not any(key.startswith("drn.") for key in state):
        raise RuntimeError(f"{checkpoint_path} does not look like a standalone single-block DRN checkpoint.")

    normalized = _normalize_legacy_single_block_state_keys(state)
    block_model.load_state_dict(normalized, strict=True)
    resistive = checkpoint.get("resistive_parameters", {})
    if resistive:
        _load_resistive_parameters(checkpoint_path, resistive, block_model)


def _normalize_legacy_single_block_state_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    legacy_to_current = {
        "drn.block.voltage_amp_raw": "drn.block._voltage_amp_raw",
        "drn.block.current_amp_raw": "drn.block._current_amp_raw",
    }
    normalized = dict(state)
    for old_key, new_key in legacy_to_current.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    return normalized


def _normalize_legacy_drn_state_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    legacy_to_current = {
        "block.voltage_amp_raw": "block._voltage_amp_raw",
        "block.current_amp_raw": "block._current_amp_raw",
    }
    normalized = dict(state)
    for old_key, new_key in legacy_to_current.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    return normalized


def _load_resistive_parameters(checkpoint_path: str | Path, resistive: dict[str, torch.Tensor], drn_mlp) -> None:
    targets = dict(drn_mlp.named_resistive_parameters())
    if set(targets) == set(resistive):
        pairs = [(name, resistive[name], targets[name]) for name in resistive]
    else:
        source_items = list(resistive.items())
        target_items = list(targets.items())
        if len(source_items) != len(target_items):
            missing = sorted(set(targets) - set(resistive))
            unexpected = sorted(set(resistive) - set(targets))
            raise RuntimeError(
                f"Could not load single-block resistive tensors from {checkpoint_path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        pairs = [
            (f"{source_name}->{target_name}", source_tensor, target_tensor)
            for (source_name, source_tensor), (target_name, target_tensor) in zip(source_items, target_items)
        ]
    with torch.no_grad():
        for name, value, target in pairs:
            if tuple(target.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"Resistive shape mismatch for {name} in {checkpoint_path}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def find_single_block_checkpoint(checkpoint_dir: str | Path, layer_index: int) -> Path:
    root = Path(checkpoint_dir)
    candidates = [
        root / f"layer_{layer_index}" / "checkpoint_best.pt",
        root / f"layer_{layer_index}" / "checkpoint_last.pt",
        root / str(layer_index) / "checkpoint_best.pt",
        root / str(layer_index) / "checkpoint_last.pt",
        root / f"layer_{layer_index}.pt",
        root / f"{layer_index}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No single-block checkpoint found for layer {layer_index} under {root}.")


def load_single_block_checkpoint_dir(model, checkpoint_dir: str | Path, layers: Iterable[int] | None = None) -> None:
    wanted = set(model.replaced_layer_indices if layers is None else [int(layer) for layer in layers])
    for layer in model.replaced_layers():
        if layer.layer_index not in wanted:
            continue
        load_single_block_checkpoint_into_layer(find_single_block_checkpoint(checkpoint_dir, layer.layer_index), layer)


def set_only_active_replaced_layer_trainable(model, active_layer_index: int | None) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in model.replaced_layers():
        active = active_layer_index is None or layer.layer_index == int(active_layer_index)
        for param in layer.drn_mlp.parameters():
            param.requires_grad_(active)
        if hasattr(layer, "drn_output_gain"):
            layer.drn_output_gain.requires_grad_(active)
        for tensor in layer.drn_mlp.resistive_param_states():
            tensor.requires_grad_(active)


def trainable_tensors(model) -> list[torch.Tensor]:
    return [tensor for tensor in _dedupe(list(model.parameters()) + model.resistive_param_states()) if tensor.requires_grad]


def _dedupe(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    seen = set()
    unique = []
    for tensor in tensors:
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tensor)
    return unique
