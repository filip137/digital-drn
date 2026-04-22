from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import Sequence

import torch.nn as nn
import torch


def _make_ff_activation(name: str | None) -> nn.Module:
    if name is None or name in {"identity", "linear", "-"}:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported ff_activation '{name}'.")


def _normalize_spec(value, count: int, name: str, default=None):
    if value is None:
        return [default] * count
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != count:
            raise ValueError(f"Expected {count} values for {name}, got {len(values)}.")
        return values
    return [value] * count


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _compute_conv_output_shape(shape, kernel_size, stride, padding):
    channels, height, width = shape
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    return channels, out_h, out_w


def _compute_pool_output_shape(shape, pool_kernel):
    channels, height, width = shape
    kernel_h, kernel_w = _pair(pool_kernel)
    out_h = (height - kernel_h) // kernel_h + 1
    out_w = (width - kernel_w) // kernel_w + 1
    return channels, out_h, out_w


class MirrorSignedDriveFrontend(nn.Module):
    """Wrap a frontend so its output is duplicated as [u, -u] along one dimension."""

    def __init__(self, base_ff: nn.Module, *, split_dim: int = 1) -> None:
        super().__init__()
        self.base_ff = base_ff
        self.split_dim = int(split_dim)

    def forward(self, x):
        base_output = self.base_ff(x)
        return torch.cat((base_output, -base_output), dim=self.split_dim)

    def optimizer_param_groups(self):
        if hasattr(self.base_ff, "optimizer_param_groups"):
            return self.base_ff.optimizer_param_groups()
        return [{"params": list(self.base_ff.parameters())}]


def wrap_mirror_signed_drive_frontend(ff: nn.Module, *, split_dim: int = 1) -> MirrorSignedDriveFrontend:
    if isinstance(ff, MirrorSignedDriveFrontend) and ff.split_dim == split_dim:
        return ff
    return MirrorSignedDriveFrontend(ff, split_dim=split_dim)


def build_default_dense_ff(
    *,
    input_dim: int,
    output_dim: int,
    ff_bias: bool,
    ff_activation: str | None,
    ff_input_shape: Sequence[int] | None,
    ff_conv_channels,
    ff_conv_kernels,
    ff_conv_strides,
    ff_conv_paddings,
    ff_conv_pool_kernels,
):
    activation = _make_ff_activation(ff_activation)

    if not ff_conv_channels:
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, output_dim, bias=ff_bias),
            activation,
        )

    if ff_input_shape is None:
        raise ValueError("ff_input_shape is required when ff_conv_channels is provided.")
    if len(ff_input_shape) != 3:
        raise ValueError("ff_input_shape must be a sequence of (channels, height, width).")

    conv_channels = list(ff_conv_channels)
    conv_kernels = _normalize_spec(ff_conv_kernels, len(conv_channels), "ff_conv_kernels", default=3)
    conv_strides = _normalize_spec(ff_conv_strides, len(conv_channels), "ff_conv_strides", default=1)
    conv_paddings = _normalize_spec(ff_conv_paddings, len(conv_channels), "ff_conv_paddings", default=0)
    conv_pools = _normalize_spec(ff_conv_pool_kernels, len(conv_channels), "ff_conv_pool_kernels", default=None)

    in_channels, height, width = ff_input_shape
    current_shape = (in_channels, height, width)
    layers = []

    for out_channels, kernel, stride, padding, pool_kernel in zip(
        conv_channels,
        conv_kernels,
        conv_strides,
        conv_paddings,
        conv_pools,
    ):
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=ff_bias,
            )
        )
        layers.append(_make_ff_activation(ff_activation))

        current_shape = _compute_conv_output_shape(
            (out_channels, current_shape[1], current_shape[2]),
            kernel,
            stride,
            padding,
        )
        in_channels = out_channels

        if pool_kernel:
            layers.append(nn.MaxPool2d(pool_kernel, stride=pool_kernel))
            current_shape = _compute_pool_output_shape(current_shape, pool_kernel)

    flattened_dim = current_shape[0] * current_shape[1] * current_shape[2]
    layers.extend(
        [
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, output_dim, bias=ff_bias),
            activation,
        ]
    )
    return nn.Sequential(*layers)
