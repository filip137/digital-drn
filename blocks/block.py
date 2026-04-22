from typing import Mapping, Sequence

import torch.nn as nn

from .base import DigitalDRNBlock
from ..energy.block_energy import DenseDRNBlockEnergy
from ..frontends import build_default_dense_ff, wrap_mirror_signed_drive_frontend


def build_dense_drn_block(
    input_dim: int,
    layer_dims: Sequence[int],
    *,
    ff: nn.Module | None = None,
    ff_bias: bool = True,
    ff_activation: str | None = "tanh",
    ff_input_shape: Sequence[int] | None = None,
    ff_conv_channels: Sequence[int] | None = None,
    ff_conv_kernels=None,
    ff_conv_strides=None,
    ff_conv_paddings=None,
    ff_conv_pool_kernels=None,
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
    quadratic_diode_param: Mapping | None = None,
    exponential_diode_param: Mapping | None = None,
    hard_sigmoid_param: Mapping | None = None,
    learn_drive_scale: bool = True,
    init_drive_scale: float = 1.0,
    drn_learning_rate: float | None = None,
) -> DigitalDRNBlock:
    """Build a dense DRN runtime block, optionally constructing a default FF frontend."""

    layer_dims = list(layer_dims)
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must contain at least one nonlinear layer and one output layer.")

    if ff is not None:
        ff_module = ff
    else:
        ff_output_dim = layer_dims[0]
        if signed_drive:
            if ff_output_dim % 2 != 0:
                raise ValueError("signed_drive=True requires an even first DRN layer width.")
            ff_output_dim = ff_output_dim // 2
        ff_module = build_default_dense_ff(
            input_dim=input_dim,
            output_dim=ff_output_dim,
            ff_bias=ff_bias,
            ff_activation=ff_activation,
            ff_input_shape=ff_input_shape,
            ff_conv_channels=ff_conv_channels,
            ff_conv_kernels=ff_conv_kernels,
            ff_conv_strides=ff_conv_strides,
            ff_conv_paddings=ff_conv_paddings,
            ff_conv_pool_kernels=ff_conv_pool_kernels,
        )
    if signed_drive:
        ff_module = wrap_mirror_signed_drive_frontend(ff_module)

    energy = DenseDRNBlockEnergy(
        layer_dims=layer_dims,
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
    )
    block = DigitalDRNBlock(
        ff=ff_module,
        energy=energy,
        ff_learning_rate=ff_learning_rate,
        num_iterations=num_iterations,
        mode=mode,
        learn_drive_scale=learn_drive_scale,
        init_drive_scale=init_drive_scale,
        drn_learning_rate=drn_learning_rate,
    )
    block.input_dim = input_dim
    block.layer_dims = tuple(layer_dims)
    block.ff_activation_name = ff_activation
    block.signed_drive = signed_drive
    return block
