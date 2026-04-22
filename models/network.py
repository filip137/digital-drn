from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ..blocks.base import DigitalDRNBlock
from ..blocks.block import build_dense_drn_block


class DigitalDRNNet(nn.Module):
    """Generic chain of Digital DRN blocks."""

    def __init__(self, blocks: Sequence[DigitalDRNBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))

    def forward(self, x: torch.Tensor, reset: bool = False, num_iterations: int | None = None):
        h = x
        for block in self.blocks:
            h = block(h, reset=reset, num_iterations=num_iterations)
        return h

    def reset_state(self, batch_size: int, device: torch.device):
        for block in self.blocks:
            block.reset_state(batch_size, device)

    def set_device(self, device: torch.device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        super().to(device)
        for block in self.blocks:
            block.set_device(device)
        return self

    def resistive_params(self):
        return [param for block in self.blocks for param in block.resistive_params()]

    def resistive_param_states(self):
        return [param.state for param in self.resistive_params()]

    def optimizer_tensors(self):
        return list(self.parameters()) + self.resistive_param_states()

    def optimizer_param_groups(self):
        groups = []
        for block in self.blocks:
            groups.extend(block.optimizer_param_groups())
        return groups

    def named_ff_parameters(self):
        for block_idx, block in enumerate(self.blocks):
            if not hasattr(block, "named_ff_parameters"):
                continue
            for name, param in block.named_ff_parameters():
                yield f"blocks.{block_idx}.{name}", param

    def named_resistive_parameters(self):
        for block_idx, block in enumerate(self.blocks):
            if not hasattr(block, "named_resistive_parameters"):
                continue
            for name, tensor in block.named_resistive_parameters():
                yield f"blocks.{block_idx}.{name}", tensor

    def enable_resistive_grad_(self, enabled: bool = True):
        for block in self.blocks:
            block.enable_resistive_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True):
        for block in self.blocks:
            block.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self):
        for block in self.blocks:
            block.clamp_resistive_params_()
        return self

    def detach_state_(self):
        for block in self.blocks:
            block.detach_state_()
        return self


class SequentialDigitalDRNNet(DigitalDRNNet):
    """Builder for a dense stack of runtime DRN blocks."""

    def __init__(
        self,
        input_dim: int,
        block_configs: Sequence[Mapping],
        **default_block_kwargs,
    ) -> None:
        blocks = []
        current_input_dim = input_dim

        for block_config in block_configs:
            config = dict(block_config)
            if "layer_dims" not in config:
                raise ValueError("Each block config must define 'layer_dims'.")
            layer_dims = config.pop("layer_dims")
            block_kwargs = {**default_block_kwargs, **config}
            block = build_dense_drn_block(
                input_dim=current_input_dim,
                layer_dims=layer_dims,
                **block_kwargs,
            )
            blocks.append(block)
            current_input_dim = list(layer_dims)[-1]

        super().__init__(blocks)
