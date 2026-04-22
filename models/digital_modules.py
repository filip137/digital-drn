from __future__ import annotations

from typing import Mapping, Sequence

import torch.nn as nn


def build_layers_from_spec(layers_spec: Sequence[Mapping]) -> nn.Sequential:
    modules = []
    for layer_spec in layers_spec:
        spec = dict(layer_spec)
        op = spec.pop("op")

        if op == "conv2d":
            modules.append(nn.Conv2d(**spec))
        elif op == "batchnorm2d":
            modules.append(nn.BatchNorm2d(**spec))
        elif op == "maxpool2d":
            modules.append(nn.MaxPool2d(**spec))
        elif op == "adaptive_avg_pool2d":
            modules.append(nn.AdaptiveAvgPool2d(**spec))
        elif op == "flatten":
            if "start_dim" not in spec:
                spec["start_dim"] = 1
            modules.append(nn.Flatten(**spec))
        elif op == "linear":
            modules.append(nn.Linear(**spec))
        elif op == "relu":
            modules.append(nn.ReLU(**spec))
        elif op == "tanh":
            modules.append(nn.Tanh())
        elif op == "identity":
            modules.append(nn.Identity())
        else:
            raise ValueError(f"Unsupported digital layer op '{op}'.")
    return nn.Sequential(*modules)


class DigitalClassifierHead(nn.Module):
    def __init__(self, module: nn.Module, learning_rate: float | None = None) -> None:
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.module(x)

    def optimizer_param_groups(self):
        params = list(self.parameters())
        if not params:
            return []
        group = {"params": params}
        if self.learning_rate is not None:
            group["lr"] = self.learning_rate
        return [group]


class CrossEntropyReadoutHead(DigitalClassifierHead):
    def __init__(
        self,
        *,
        num_classes: int,
        in_features: int | None = None,
        flatten: bool = False,
        learning_rate: float | None = None,
    ) -> None:
        modules: list[nn.Module] = []
        if flatten:
            modules.append(nn.Flatten(1))
        if in_features is None:
            modules.append(nn.LazyLinear(num_classes))
        else:
            modules.append(nn.Linear(in_features, num_classes))

        module: nn.Module
        if len(modules) == 1:
            module = modules[0]
        else:
            module = nn.Sequential(*modules)

        super().__init__(module=module, learning_rate=learning_rate)


def build_head_from_spec(head_cfg: Mapping) -> DigitalClassifierHead:
    head_type = head_cfg.get("type", "digital_classifier_head")
    learning_rate = head_cfg.get("lr", head_cfg.get("learning_rate"))

    if head_type == "digital_classifier_head":
        return DigitalClassifierHead(
            build_layers_from_spec(head_cfg.get("layers", [])),
            learning_rate=learning_rate,
        )

    if head_type == "cross_entropy_readout":
        num_classes = head_cfg.get("num_classes", head_cfg.get("out_features"))
        if num_classes is None:
            raise ValueError("cross_entropy_readout head requires num_classes or out_features.")
        return CrossEntropyReadoutHead(
            num_classes=int(num_classes),
            in_features=head_cfg.get("in_features"),
            flatten=bool(head_cfg.get("flatten", False)),
            learning_rate=learning_rate,
        )

    raise ValueError(f"Unsupported head.type '{head_type}'.")


class DigitalModuleBlock(nn.Module):
    def __init__(self, module: nn.Module, learning_rate: float | None = None) -> None:
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate

    def forward(self, x, reset: bool = False, num_iterations: int | None = None):
        del reset, num_iterations
        return self.module(x)

    def set_device(self, device):
        self.module.to(device)
        return self

    def resistive_params(self):
        return []

    def ff_parameters(self):
        return list(self.module.parameters())

    def named_ff_parameters(self):
        yield from self.module.named_parameters()

    def named_resistive_parameters(self):
        if False:
            yield None, None

    def optimizer_param_groups(self):
        params = self.ff_parameters()
        if not params:
            return []
        group = {"params": params}
        if self.learning_rate is not None:
            group["lr"] = self.learning_rate
        return [group]

    def enable_resistive_grad_(self, enabled: bool = True):
        del enabled
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True):
        del set_to_none
        return self

    def clamp_resistive_params_(self):
        return self

    def detach_state_(self):
        return self

    def reset_state(self, batch_size: int, device):
        del batch_size, device
        return self
