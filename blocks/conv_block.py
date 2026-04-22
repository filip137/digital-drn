from __future__ import annotations

from typing import Mapping, Sequence

import torch.nn as nn

from .base import DigitalDRNBlock
from ..energy.block_energy import ConvDRNBlockEnergy, ConvDenseDRNBlockEnergy
from ..frontends import wrap_mirror_signed_drive_frontend


def build_conv_drn_block(
    ff: nn.Module,
    layer_shapes: Sequence[Sequence[int]],
    *,
    kernel_sizes,
    strides=1,
    paddings=0,
    dilations=1,
    ff_learning_rate: float | None = None,
    signed_drive: bool = False,
    num_iterations: int = 6,
    mode: str = "asynchronous",
    non_linearity: str = "linear",
    output_non_linearity: str | None = None,
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
    """Build a conv-state DRN runtime block."""

    layer_shapes = tuple(tuple(shape) for shape in layer_shapes)
    ff_module = wrap_mirror_signed_drive_frontend(ff) if signed_drive else ff
    energy = ConvDRNBlockEnergy(
        layer_shapes=layer_shapes,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        dilations=dilations,
        non_linearity=non_linearity,
        output_non_linearity=output_non_linearity,
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
    block.layer_shapes = layer_shapes
    block.signed_drive = signed_drive
    return block


def build_conv_dense_drn_block(
    ff: nn.Module,
    conv_state_shape: Sequence[int],
    output_dim: int,
    *,
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
    """Build a mixed conv+dense DRN runtime block."""

    conv_state_shape = tuple(conv_state_shape)
    output_dim = int(output_dim)
    ff_module = wrap_mirror_signed_drive_frontend(ff) if signed_drive else ff
    energy = ConvDenseDRNBlockEnergy(
        conv_state_shape=conv_state_shape,
        output_dim=output_dim,
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
    block.conv_state_shape = conv_state_shape
    block.output_dim = output_dim
    block.signed_drive = signed_drive
    return block
