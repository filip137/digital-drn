from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..blocks.base import DigitalDRNBlock
from ..blocks.block import build_dense_drn_block


class TokenwiseDRNMLP(nn.Module):
    """Transformer MLP replacement that applies one shared DRN block per token.

    Inputs are expected to have shape ``(..., d_model)``. All leading dimensions
    are treated as an independent sample axis, so a tensor shaped
    ``(batch, seq_len, d_model)`` is processed token-by-token with shared DRN
    parameters and reshaped back to the original leading dimensions.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int | None = None,
        layer_dims: Sequence[int] | None = None,
        block: DigitalDRNBlock | None = None,
        dropout: float = 0.0,
        reset_each_forward: bool = True,
        ff: nn.Module | None = None,
        ff_bias: bool = True,
        ff_activation: str | None = "identity",
        ff_learning_rate: float | None = None,
        signed_drive: bool = False,
        num_iterations: int = 6,
        mode: str = "asynchronous",
        non_linearity: str = "linear",
        weight_gains=0.1,
        bias_gain: float = 0.0,
        voltage_amp: float = 1.0,
        current_amp: float = 1.0,
        amplify_first_free_layer: bool = True,
        weight_min=None,
        weight_max=None,
        weight_init_mode: str = "kaiming_uniform",
        quadratic_diode_param: dict | None = None,
        exponential_diode_param: dict | None = None,
        hard_sigmoid_param: dict | None = None,
        learn_drive_scale: bool = True,
        init_drive_scale: float = 1.0,
        drn_learning_rate: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be strictly positive.")
        if block is not None and layer_dims is not None:
            raise ValueError("Specify either block or layer_dims, not both.")
        if block is not None and hidden_dim is not None:
            raise ValueError("Specify either block or hidden_dim, not both.")
        if block is not None and ff is not None:
            raise ValueError("Specify either block or ff, not both.")

        self.d_model = int(d_model)
        self.reset_each_forward = bool(reset_each_forward)

        if block is None:
            if layer_dims is None:
                resolved_hidden_dim = int(hidden_dim if hidden_dim is not None else 4 * d_model)
                layer_dims = [resolved_hidden_dim, d_model]
            else:
                layer_dims = list(layer_dims)
                if not layer_dims:
                    raise ValueError("layer_dims must not be empty.")
                if layer_dims[-1] != d_model:
                    raise ValueError("The final DRN layer dimension must match d_model.")

            block = build_dense_drn_block(
                input_dim=d_model,
                layer_dims=layer_dims,
                ff=ff,
                ff_bias=ff_bias,
                ff_activation=ff_activation,
                ff_learning_rate=ff_learning_rate,
                signed_drive=signed_drive,
                num_iterations=num_iterations,
                mode=mode,
                non_linearity=non_linearity,
                weight_gains=weight_gains,
                bias_gain=bias_gain,
                voltage_amp=voltage_amp,
                current_amp=current_amp,
                amplify_first_free_layer=amplify_first_free_layer,
                weight_min=weight_min,
                weight_max=weight_max,
                weight_init_mode=weight_init_mode,
                quadratic_diode_param=quadratic_diode_param,
                exponential_diode_param=exponential_diode_param,
                hard_sigmoid_param=hard_sigmoid_param,
                learn_drive_scale=learn_drive_scale,
                init_drive_scale=init_drive_scale,
                drn_learning_rate=drn_learning_rate,
            )
        elif getattr(block, "layer_dims", None) is not None and tuple(block.layer_dims)[-1] != d_model:
            raise ValueError("The supplied block output dimension must match d_model.")

        self.block = block
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        reset: bool | None = None,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("TokenwiseDRNMLP expects an input shaped (..., d_model).")
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected trailing dimension {self.d_model}, got {x.size(-1)}.")

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d_model)
        y_flat = self.block(
            x_flat,
            reset=self.reset_each_forward if reset is None else bool(reset),
            num_iterations=num_iterations,
        )
        y = y_flat.reshape(*leading_shape, self.d_model)
        return self.dropout(y)

    def set_device(self, device: torch.device | str):
        self.block.set_device(torch.device(device))
        self.dropout.to(device)
        return self

    def _apply(self, fn):
        super()._apply(fn)
        # Keep non-registered DRN energy tensors aligned when callers use module.to(...).
        self.block._sync_energy_device(self.block._drive_scale_raw.device)
        return self

    def ff_parameters(self):
        return self.block.ff_parameters()

    def resistive_params(self):
        return self.block.resistive_params()

    def resistive_param_states(self):
        return self.block.resistive_param_states()

    def optimizer_tensors(self):
        return list(self.parameters()) + self.resistive_param_states()

    def optimizer_param_groups(self):
        return self.block.optimizer_param_groups()

    def named_ff_parameters(self):
        yield from self.block.named_ff_parameters()

    def named_resistive_parameters(self):
        yield from self.block.named_resistive_parameters()

    def enable_resistive_grad_(self, enabled: bool = True):
        self.block.enable_resistive_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True):
        self.block.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self):
        self.block.clamp_resistive_params_()
        return self

    def detach_state_(self):
        self.block.detach_state_()
        return self

    def collect_diagnostics(self):
        return self.block.collect_diagnostics()
