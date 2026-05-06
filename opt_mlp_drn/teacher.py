from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TeacherLayerActivations:
    """Teacher-forced tensors for one OPT decoder MLP layer.

    ``z`` is the normalized MLP input, ``r`` is the teacher MLP delta,
    ``a`` is the post-attention residual before the MLP residual update, and
    ``h_next`` is the teacher block output after adding the MLP residual.
    """

    z: torch.Tensor
    r: torch.Tensor
    a: torch.Tensor
    h_next: torch.Tensor
    next_ln: torch.Tensor | None = None


def default_probe_layers(num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    return sorted({0, num_layers // 2, num_layers - 1})


def opt_num_layers(model: nn.Module) -> int:
    return len(_decoder_layers(model))


@torch.no_grad()
def collect_teacher_layer_activations(
    teacher: nn.Module,
    input_ids: torch.LongTensor,
    layer_indices: list[int],
    *,
    attention_mask: torch.Tensor | None = None,
    mlp_input_mode: str = "normalized",
) -> dict[int, TeacherLayerActivations]:
    """Run OPT under teacher forcing and collect per-layer MLP targets."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if mlp_input_mode not in {"normalized", "raw"}:
        raise ValueError("mlp_input_mode must be either 'normalized' or 'raw'.")

    layers = _decoder_layers(teacher)
    bad = [idx for idx in layer_indices if idx < 0 or idx >= len(layers)]
    if bad:
        raise ValueError(f"Layer indices out of range for {len(layers)} OPT layers: {bad}.")

    was_training = teacher.training
    teacher.eval()
    records: dict[int, dict[str, torch.Tensor]] = {idx: {} for idx in layer_indices}
    handles = []

    for idx in layer_indices:
        layer = layers[idx]

        def fc1_pre_hook(module: nn.Module, inputs: tuple[Any, ...], layer_idx: int = idx) -> None:
            del module
            records[layer_idx]["z_normalized"] = _as_btd(inputs[0], input_ids.shape).detach()

        def fc2_hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
            layer_idx: int = idx,
        ) -> None:
            del module, inputs
            records[layer_idx]["r_normalized"] = _as_btd(output, input_ids.shape).detach()

        def final_ln_hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
            layer_idx: int = idx,
            opt_layer: nn.Module = layer,
        ) -> None:
            del module, output
            if bool(getattr(opt_layer, "do_layer_norm_before", True)):
                records[layer_idx]["a"] = _as_btd(inputs[0], input_ids.shape).detach()

        def layer_hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            layer_idx: int = idx,
        ) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            records[layer_idx]["h_next"] = _as_btd(hidden, input_ids.shape).detach()

        handles.append(layer.fc1.register_forward_pre_hook(fc1_pre_hook))
        handles.append(layer.fc2.register_forward_hook(fc2_hook))
        handles.append(layer.final_layer_norm.register_forward_hook(final_ln_hook))
        handles.append(layer.register_forward_hook(layer_hook))

    try:
        teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            teacher.train()

    result: dict[int, TeacherLayerActivations] = {}
    for idx in layer_indices:
        record = records[idx]
        missing = {"z_normalized", "r_normalized", "h_next"} - set(record)
        if missing:
            raise RuntimeError(f"Missing teacher activation(s) for layer {idx}: {sorted(missing)}.")
        if "a" not in record:
            # OPT-125M is pre-LN and should hit the final_layer_norm hook above.
            # This fallback keeps tiny/custom OPT configs usable.
            record["a"] = record["h_next"] - record["r_normalized"]
        if mlp_input_mode == "raw":
            z = record["a"]
            r = _teacher_mlp_delta(layers[idx], z).detach()
            h_next = (z + r).detach()
        else:
            z = record["z_normalized"]
            r = record["r_normalized"]
            h_next = record["h_next"]
        result[idx] = TeacherLayerActivations(
            z=z,
            r=r,
            a=record["a"],
            h_next=h_next,
            next_ln=apply_next_layer_norm(teacher, idx, h_next).detach(),
        )
    return result


def apply_next_layer_norm(teacher: nn.Module, layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply LN1 of layer ``layer_index + 1`` or the decoder final LN at the end."""

    decoder = _decoder(teacher)
    layers = list(decoder.layers)
    if layer_index + 1 < len(layers):
        return layers[layer_index + 1].self_attn_layer_norm(hidden_states)
    final_ln = getattr(decoder, "final_layer_norm", None)
    if final_ln is None:
        return hidden_states
    return final_ln(hidden_states)


def _decoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "base"):
        model = model.base
    return model.model.decoder


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    return list(_decoder(model).layers)


def _teacher_mlp_delta(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = layer.fc1(hidden_states)
    hidden_states = layer.activation_fn(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    return hidden_states


def _as_btd(tensor: torch.Tensor, input_shape: torch.Size | tuple[int, int]) -> torch.Tensor:
    batch, seq_len = int(input_shape[0]), int(input_shape[1])
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D OPT activation, got shape {tuple(tensor.shape)}.")
    if tensor.size(0) != batch * seq_len:
        raise ValueError(
            f"Flat activation has first dimension {tensor.size(0)}, expected {batch * seq_len}."
        )
    return tensor.view(batch, seq_len, tensor.size(-1))
