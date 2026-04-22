from __future__ import annotations

import torch

from .network import DigitalDRNNet


class DigitalAnalogNet(DigitalDRNNet):
    def __init__(self, blocks, head=None) -> None:
        super().__init__(blocks)
        self.head = head

    def forward(self, x: torch.Tensor, reset: bool = False, num_iterations: int | None = None):
        h = super().forward(x, reset=reset, num_iterations=num_iterations)
        if self.head is not None:
            h = self.head(h)
        return h

    def set_device(self, device: torch.device):
        super().set_device(device)
        if self.head is not None:
            self.head.to(device)
        return self

    def optimizer_param_groups(self):
        groups = super().optimizer_param_groups()
        if self.head is not None:
            if hasattr(self.head, "optimizer_param_groups"):
                groups.extend(self.head.optimizer_param_groups())
            else:
                params = list(self.head.parameters())
                if params:
                    groups.append({"params": params})
        return groups
