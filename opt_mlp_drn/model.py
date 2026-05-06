from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .drn_architecture import (
    build_tokenwise_drn_mlp,
    resolve_drn_layer_dims,
    resolve_logical_hidden_dim,
)


def load_opt_causal_lm(model_name: str = "facebook/opt-125m") -> nn.Module:
    """Load an OPT causal LM, including cached legacy .bin checkpoints."""

    try:
        from transformers import OPTForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to load Hugging Face OPT checkpoints.") from exc

    try:
        return OPTForCausalLM.from_pretrained(model_name)
    except ValueError as exc:
        message = str(exc)
        if "torch.load" not in message or "safetensors" not in message:
            raise
        return _load_opt_pytorch_bin_safely(model_name)


@dataclass
class LayerDistillationCache:
    teacher_delta: torch.Tensor
    student_delta: torch.Tensor
    teacher_post_residual: torch.Tensor
    student_post_residual: torch.Tensor


class OPTDRNDecoderLayer(nn.Module):
    """OPT decoder layer with the MLP residual update replaced by a DRN."""

    def __init__(
        self,
        original_layer: nn.Module,
        *,
        layer_index: int,
        drn_iter: int = 4,
        signed_drive: bool = True,
        drive_architecture: str = "projected_hidden",
        non_linearity: str = "perfect_diode",
        hidden_multiplier: float | None = None,
        dropout: float | None = None,
        weight_gains: float = 0.1,
        weight_min: float | None = 1.0e-5,
        weight_max: float | None = None,
        hard_sigmoid_param: dict[str, float] | None = None,
        bias_gain: float = 0.0,
        signed_output_weights: bool = False,
        init_drive_scale: float = 1.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        learn_amplification: bool = False,
        drn_input_scale: float = 1.0,
        drn_output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.embed_dim = int(original_layer.embed_dim)
        self.dropout = float(original_layer.dropout if dropout is None else dropout)
        self.do_layer_norm_before = bool(original_layer.do_layer_norm_before)

        self.self_attn = original_layer.self_attn
        self.self_attn_layer_norm = original_layer.self_attn_layer_norm
        self.final_layer_norm = original_layer.final_layer_norm

        self.fc1 = original_layer.fc1
        self.fc2 = original_layer.fc2
        self.activation_fn = original_layer.activation_fn
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

        logical_hidden_dim = resolve_logical_hidden_dim(
            embed_dim=self.embed_dim,
            teacher_hidden_dim=int(self.fc1.out_features),
            hidden_multiplier=hidden_multiplier,
        )
        self.drn_drive_architecture = str(drive_architecture)
        self.drn_layer_dims = resolve_drn_layer_dims(
            embed_dim=self.embed_dim,
            logical_hidden_dim=logical_hidden_dim,
            signed_drive=signed_drive,
            drive_architecture=self.drn_drive_architecture,
        )

        self.drn_mlp = build_tokenwise_drn_mlp(
            d_model=self.embed_dim,
            layer_dims=self.drn_layer_dims,
            drive_architecture=self.drn_drive_architecture,
            signed_drive=bool(signed_drive),
            signed_output_weights=bool(signed_output_weights),
            dropout=self.dropout,
            drn_iter=int(drn_iter),
            non_linearity=str(non_linearity),
            weight_gains=float(weight_gains),
            bias_gain=float(bias_gain),
            voltage_amp=float(voltage_amp),
            current_amp=float(current_amp),
            learn_amplification=bool(learn_amplification),
            weight_min=weight_min,
            weight_max=weight_max,
            hard_sigmoid_param=hard_sigmoid_param,
            learn_drive_scale=True,
            init_drive_scale=float(init_drive_scale),
        )
        self.drn_mlp.enable_resistive_grad_()
        self.register_buffer("drn_input_scale", torch.tensor(float(max(drn_input_scale, 1.0e-12))))
        self.register_buffer("drn_output_scale", torch.tensor(float(max(drn_output_scale, 1.0e-12))))
        self.drn_output_gain = nn.Parameter(torch.tensor(1.0))
        self._distill_cache: LayerDistillationCache | None = None

    def clear_distillation_cache(self) -> None:
        self._distill_cache = None

    def distillation_cache(self) -> LayerDistillationCache:
        if self._distill_cache is None:
            raise RuntimeError(f"Layer {self.layer_index} has no distillation cache.")
        return self._distill_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del use_cache
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states_shape = hidden_states.shape
        pre_mlp_residual = hidden_states.reshape(-1, hidden_states.size(-1))
        mlp_input = pre_mlp_residual
        if self.do_layer_norm_before:
            mlp_input = self.final_layer_norm(mlp_input)

        mlp_input_for_drn = mlp_input.view(*hidden_states_shape)
        scaled_mlp_input = mlp_input_for_drn / self.drn_input_scale.clamp_min(1.0e-12)
        student_delta = self.drn_output_gain * self.drn_output_scale * self.drn_mlp(scaled_mlp_input, reset=True)
        student_delta_flat = student_delta.reshape(-1, student_delta.size(-1))
        student_post = pre_mlp_residual + student_delta_flat

        with torch.no_grad():
            teacher_delta = self.teacher_mlp_delta(mlp_input_for_drn)
            teacher_delta_flat = teacher_delta.reshape(-1, teacher_delta.size(-1))
            teacher_post = pre_mlp_residual.detach() + teacher_delta_flat
            if not self.do_layer_norm_before:
                teacher_post = self.final_layer_norm(teacher_post)

        if not self.do_layer_norm_before:
            student_post = self.final_layer_norm(student_post)

        self._distill_cache = LayerDistillationCache(
            teacher_delta=teacher_delta.detach(),
            student_delta=student_delta,
            teacher_post_residual=teacher_post.view(hidden_states_shape).detach(),
            student_post_residual=student_post.view(hidden_states_shape),
        )
        return student_post.view(hidden_states_shape)

    @torch.no_grad()
    def teacher_mlp_delta(self, normalized_hidden_states: torch.Tensor) -> torch.Tensor:
        shape = normalized_hidden_states.shape
        x = normalized_hidden_states.reshape(-1, normalized_hidden_states.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x.view(shape)

    def drn_mlp_delta(
        self,
        normalized_hidden_states: torch.Tensor,
        *,
        reset: bool | None = None,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        scaled = normalized_hidden_states / self.drn_input_scale.clamp_min(1.0e-12)
        return self.drn_output_gain * self.drn_output_scale * self.drn_mlp(scaled, reset=reset, num_iterations=num_iterations)

    def set_drn_scales(self, input_scale: float | torch.Tensor, output_scale: float | torch.Tensor) -> None:
        self.drn_input_scale.copy_(torch.as_tensor(input_scale, device=self.drn_input_scale.device).float())
        self.drn_output_scale.copy_(torch.as_tensor(output_scale, device=self.drn_output_scale.device).float())

    def set_drn_output_gain(self, output_gain: float | torch.Tensor) -> None:
        with torch.no_grad():
            self.drn_output_gain.copy_(torch.as_tensor(output_gain, device=self.drn_output_gain.device).float())


class OPTMLPDRNForCausalLM(nn.Module):
    """Hugging Face OPT causal LM with selected MLP layers replaced by DRNs."""

    def __init__(
        self,
        base: nn.Module,
        *,
        replace_mlp_layers: str | Iterable[int] = "last:1",
        freeze_base: bool = True,
        drn_iter: int = 4,
        signed_drive: bool = True,
        drive_architecture: str = "projected_hidden",
        non_linearity: str = "perfect_diode",
        hidden_multiplier: float | None = None,
        weight_gains: float = 0.1,
        weight_min: float | None = 1.0e-5,
        weight_max: float | None = None,
        hard_sigmoid_param: dict[str, float] | None = None,
        bias_gain: float = 0.0,
        signed_output_weights: bool = False,
        init_drive_scale: float = 1.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        learn_amplification: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.freeze_base = bool(freeze_base)
        self.hf_config = base.config
        self.config = _compat_config(base.config)

        layers = self.base.model.decoder.layers
        if self.freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.replaced_layer_indices = parse_layer_indices(replace_mlp_layers, len(layers))
        for index in self.replaced_layer_indices:
            original_layer = layers[index]
            layers[index] = OPTDRNDecoderLayer(
                original_layer,
                layer_index=index,
                drn_iter=drn_iter,
                signed_drive=signed_drive,
                drive_architecture=drive_architecture,
                non_linearity=non_linearity,
                hidden_multiplier=hidden_multiplier,
                dropout=getattr(base.config, "dropout", None),
                weight_gains=weight_gains,
                weight_min=weight_min,
                weight_max=weight_max,
                hard_sigmoid_param=hard_sigmoid_param,
                bias_gain=bias_gain,
                signed_output_weights=signed_output_weights,
                init_drive_scale=init_drive_scale,
                voltage_amp=voltage_amp,
                current_amp=current_amp,
                learn_amplification=learn_amplification,
            )

        self.enable_resistive_grad_(True)
        self.last_diagnostics: dict[str, float | int | str | list[int] | bool] = {
            "model_type": "opt_mlp_drn",
            "replaced_layer_indices": list(self.replaced_layer_indices),
            "freeze_base": self.freeze_base,
            "drn_signed_drive": bool(signed_drive),
            "drn_drive_architecture": str(drive_architecture),
            "drn_signed_output_weights": bool(signed_output_weights),
            "drn_iter": int(drn_iter),
            "drn_non_linearity": str(non_linearity),
            "drn_weight_min": weight_min,
            "drn_weight_max": weight_max,
            "hard_sigmoid_param": hard_sigmoid_param,
            "drn_voltage_amp_init": float(voltage_amp),
            "drn_current_amp_init": float(current_amp),
            "drn_learn_amplification": bool(learn_amplification),
        }
        if self.freeze_base:
            self.base.eval()

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/opt-125m", **kwargs: Any) -> "OPTMLPDRNForCausalLM":
        base = load_opt_causal_lm(model_name)
        return cls(base, **kwargs)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "OPTMLPDRNForCausalLM":
        try:
            from transformers import OPTForCausalLM
        except ImportError as exc:
            raise ImportError("Install transformers to construct OPT models.") from exc

        base = OPTForCausalLM(config)
        return cls(base, **kwargs)

    def train(self, mode: bool = True) -> "OPTMLPDRNForCausalLM":
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
            for mlp in self.drn_mlps():
                mlp.train(mode)
        return self

    def replaced_layers(self) -> list[OPTDRNDecoderLayer]:
        return [
            layer
            for layer in self.base.model.decoder.layers
            if isinstance(layer, OPTDRNDecoderLayer)
        ]

    def drn_mlps(self) -> list[TokenwiseDRNMLP]:
        return [layer.drn_mlp for layer in self.replaced_layers()]

    def clear_distillation_caches(self) -> None:
        for layer in self.replaced_layers():
            layer.clear_distillation_cache()

    def resistive_param_states(self) -> list[torch.Tensor]:
        return [state for mlp in self.drn_mlps() for state in mlp.resistive_param_states()]

    def named_resistive_parameters(self):
        for layer in self.replaced_layers():
            for name, tensor in layer.drn_mlp.named_resistive_parameters():
                yield f"base.model.decoder.layers.{layer.layer_index}.drn_mlp.{name}", tensor

    def optimizer_tensors(self) -> list[torch.Tensor]:
        return list(self.parameters()) + self.resistive_param_states()

    def enable_resistive_grad_(self, enabled: bool = True) -> "OPTMLPDRNForCausalLM":
        for mlp in self.drn_mlps():
            mlp.enable_resistive_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True) -> "OPTMLPDRNForCausalLM":
        for mlp in self.drn_mlps():
            mlp.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self) -> "OPTMLPDRNForCausalLM":
        for mlp in self.drn_mlps():
            mlp.clamp_resistive_params_()
        return self

    def detach_state_(self) -> "OPTMLPDRNForCausalLM":
        for mlp in self.drn_mlps():
            mlp.detach_state_()
        return self

    def collect_diagnostics(self) -> dict[str, float | int | str | list[int] | bool]:
        diagnostics = dict(self.last_diagnostics)
        drive_scales = [float(mlp.block.drive_scale.detach().item()) for mlp in self.drn_mlps()]
        if drive_scales:
            diagnostics["mean_drive_scale"] = float(sum(drive_scales) / len(drive_scales))
        for layer in self.replaced_layers():
            idx = layer.layer_index
            mlp = layer.drn_mlp
            diagnostics[f"layer_{idx}/output_gain"] = float(layer.drn_output_gain.detach().item())
            try:
                mlp_diagnostics = mlp.collect_diagnostics()
            except RuntimeError:
                continue
            for key, value in mlp_diagnostics.items():
                diagnostics[f"layer_{idx}/{key}"] = value
        return diagnostics

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        return_hidden_states: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        captured_hidden: list[torch.Tensor] = []
        handles = []
        if return_hidden_states:
            layers = list(self.base.model.decoder.layers)

            def capture_initial(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not captured_hidden and inputs:
                    captured_hidden.append(inputs[0])

            def capture_layer(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                captured_hidden.append(tensor)

            if layers:
                handles.append(layers[0].register_forward_pre_hook(capture_initial))
                handles.extend(layer.register_forward_hook(capture_layer) for layer in layers)
        try:
            out = self.base(
                input_ids=input_ids,
                labels=None,
                output_hidden_states=return_hidden_states,
                return_dict=True,
                **kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()
        logits = out.logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        hidden_states = None
        if return_hidden_states:
            hidden_states = captured_hidden
            if out.hidden_states is not None and hidden_states:
                hidden_states[-1] = out.hidden_states[-1]
        return {"logits": logits, "loss": loss, "hidden_states": hidden_states}

    def distillation_loss(self, input_ids: torch.LongTensor) -> dict[str, torch.Tensor | dict[str, float]]:
        self.clear_distillation_caches()
        self(input_ids, targets=None)

        losses = []
        delta_losses = []
        post_residual_losses = []
        final_layer_index = len(self.base.model.decoder.layers) - 1
        final_ln = self.base.model.decoder.final_layer_norm

        for layer in self.replaced_layers():
            cache = layer.distillation_cache()
            student_target = cache.student_post_residual
            teacher_target = cache.teacher_post_residual
            if layer.layer_index == final_layer_index and final_ln is not None:
                student_target = final_ln(student_target)
                with torch.no_grad():
                    teacher_target = final_ln(teacher_target)
            loss = F.mse_loss(student_target, teacher_target.detach())
            losses.append(loss)
            delta_losses.append(F.mse_loss(cache.student_delta, cache.teacher_delta.detach()))
            post_residual_losses.append(
                F.mse_loss(cache.student_post_residual, cache.teacher_post_residual.detach())
            )

        if not losses:
            raise RuntimeError("No replaced MLP layers are available for distillation.")

        distill_loss = torch.stack(losses).mean()
        delta_mse = torch.stack(delta_losses).mean()
        post_residual_mse = torch.stack(post_residual_losses).mean()
        metrics = {
            "distill_loss": float(distill_loss.detach().item()),
            "delta_mse": float(delta_mse.detach().item()),
            "post_residual_mse": float(post_residual_mse.detach().item()),
        }
        metrics.update(self.collect_diagnostics())
        return {"loss": distill_loss, "metrics": metrics}


def parse_layer_indices(raw: str | Iterable[int], num_layers: int) -> list[int]:
    """Parse zero-based OPT layer indices.

    Accepted string forms: ``all``, ``none``, ``last:N``, ``first:N``, and
    comma-separated indices/ranges such as ``0,3,6-11``.
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be strictly positive.")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text == "all":
            return list(range(num_layers))
        if text in {"none", ""}:
            return []
        indices: list[int] = []
        for part in (item.strip() for item in text.split(",")):
            if not part:
                continue
            if part.startswith("last:"):
                count = _parse_nonnegative_count(part, "last:")
                indices.extend(range(max(0, num_layers - count), num_layers))
                continue
            if part.startswith("first:"):
                count = _parse_nonnegative_count(part, "first:")
                indices.extend(range(0, min(count, num_layers)))
                continue
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise ValueError(f"Invalid descending layer range '{part}'.")
                indices.extend(range(start, end + 1))
                continue
            indices.append(int(part))
    else:
        indices = [int(index) for index in raw]

    bad = [index for index in indices if index < 0 or index >= num_layers]
    if bad:
        raise ValueError(f"Layer indices must be zero-based in [0, {num_layers - 1}], got {bad}.")
    return sorted(set(indices))


def _parse_nonnegative_count(part: str, prefix: str) -> int:
    count = int(part[len(prefix) :])
    if count < 0:
        raise ValueError(f"{prefix} count must be non-negative.")
    return count


def _compat_config(hf_config: Any) -> Any:
    block_size = int(getattr(hf_config, "max_position_embeddings", 2048))
    return SimpleNamespace(
        block_size=block_size,
        vocab_size=int(getattr(hf_config, "vocab_size")),
        n_layer=int(getattr(hf_config, "num_hidden_layers")),
        n_head=int(getattr(hf_config, "num_attention_heads")),
        n_embd=int(getattr(hf_config, "hidden_size")),
        model_type="opt",
    )


def _load_opt_pytorch_bin_safely(model_name: str) -> nn.Module:
    """Load OPT legacy .bin checkpoints when Transformers blocks torch<2.6."""

    try:
        from transformers import OPTConfig, OPTForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to load Hugging Face OPT checkpoints.") from exc

    model_dir = Path(model_name)
    if not model_dir.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub to resolve cached OPT checkpoints.") from exc
        model_dir = Path(
            snapshot_download(
                repo_id=model_name,
                allow_patterns=["config.json", "pytorch_model.bin"],
                local_files_only=True,
            )
        )

    weight_path = model_dir / "pytorch_model.bin"
    if not weight_path.exists():
        raise FileNotFoundError(f"Could not find {weight_path}.")

    config = OPTConfig.from_pretrained(str(model_dir))
    base = OPTForCausalLM(config)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state_dict, strict=True)
    return base
