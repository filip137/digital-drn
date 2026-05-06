import torch
import torch.nn as nn
import pytest

from digital_drn import DenseDRNBlockEnergy, MirrorSignedDriveFrontend
from digital_drn.blocks import (
    DigitalDRNBlock,
    build_conv_dense_drn_block,
    build_conv_drn_block,
    build_dense_drn_block,
)


def test_block_forward_reset_and_iteration_override():
    ff = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(3, 4, bias=False))
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        ff=ff,
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    with torch.no_grad():
        block.ff[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        block.energy.dense_weights[0].state.fill_(0.5)
        block.energy.biases[0].state.zero_()

    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=torch.float32)

    y = block(x, reset=True, num_iterations=2)
    assert y.shape == (2, 2)
    assert not torch.allclose(y, torch.zeros_like(y))

    y_hold = block(x, reset=False, num_iterations=0)
    assert torch.allclose(y_hold, y)

    y_reset = block(x, reset=True, num_iterations=0)
    assert torch.allclose(y_reset, torch.zeros_like(y_reset))
    assert block.minimizer.num_iterations == 2


def test_default_ff_path_includes_requested_activation():
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        ff_activation="relu",
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    assert isinstance(block.ff[-1], nn.ReLU)


def test_signed_drive_wraps_default_dense_frontend():
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        signed_drive=True,
        ff_activation="identity",
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    assert isinstance(block.ff, MirrorSignedDriveFrontend)
    assert isinstance(block.ff.base_ff[-1], nn.Identity)

    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    u = block.ff.base_ff(x)
    drive = block.ff(x)
    assert drive.shape[1] == 2 * u.shape[1]
    assert torch.allclose(drive[:, : u.shape[1]], u)
    assert torch.allclose(drive[:, u.shape[1] :], -u)


def test_conv_ff_path_produces_block_output():
    block = build_dense_drn_block(
        input_dim=28 * 28,
        layer_dims=[8, 2],
        ff_activation="tanh",
        ff_input_shape=(1, 28, 28),
        ff_conv_channels=[4, 8],
        ff_conv_kernels=[3, 3],
        ff_conv_paddings=[1, 1],
        ff_conv_pool_kernels=[2, 2],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    x = torch.randn(2, 1, 28, 28)
    y = block(x, reset=True, num_iterations=1)

    assert isinstance(block.ff[0], nn.Conv2d)
    assert y.shape == (2, 2)


def test_dense_and_conv_block_wrappers_share_runtime_base():
    dense_block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )
    conv_block = build_conv_drn_block(
        ff=nn.Identity(),
        layer_shapes=[(2, 4, 4)],
        kernel_sizes=3,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[],
        bias_gain=0.0,
    )
    conv_dense_block = build_conv_dense_drn_block(
        ff=nn.Identity(),
        conv_state_shape=(2, 4, 4),
        output_dim=3,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    for block in (dense_block, conv_block, conv_dense_block):
        assert isinstance(block, DigitalDRNBlock)
        assert block.minimizer is block.inference_minimizer
        assert block.augmented_minimizer is block.training_minimizer


def test_runtime_block_respects_custom_ff_optimizer_groups():
    class GroupedFF(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, x):
            return self.linear(x)

        def optimizer_param_groups(self):
            return [
                {"params": [self.linear.weight], "lr": 0.02},
                {"params": [self.linear.bias]},
            ]

    block = DigitalDRNBlock(
        ff=GroupedFF(),
        energy=DenseDRNBlockEnergy(
            layer_dims=[4, 2],
            non_linearity="linear",
            weight_gains=[0.1],
            bias_gain=0.0,
        ),
        ff_learning_rate=0.01,
        drn_learning_rate=0.001,
    )

    groups = block.optimizer_param_groups()

    assert len(groups) == 4
    assert groups[0]["params"] == [block.ff.linear.weight]
    assert groups[0]["lr"] == 0.02
    assert groups[1]["params"] == [block.ff.linear.bias]
    assert groups[1]["lr"] == 0.01
    assert groups[2]["params"] == [block._drive_scale_raw]
    assert groups[2]["lr"] == 0.01
    assert all(group["lr"] == 0.001 for group in groups[3:])


def test_runtime_block_can_learn_amplification():
    torch.manual_seed(11)
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        hard_sigmoid_param={"g_on": 10.0, "g_off": 1.0e-7, "v_min": -1.2, "v_max": 1.2},
        weight_gains=[0.1],
        bias_gain=0.0,
        voltage_amp=1.0,
        current_amp=1.0,
        learn_voltage_amp=True,
        learn_current_amp=True,
    )

    x = torch.randn(3, 3)
    y = block(x, reset=True)
    loss = y.pow(2).mean()
    loss.backward()

    amp_params = dict(block.named_amplification_parameters())
    assert amp_params["voltage_amp_raw"].grad is not None
    assert amp_params["current_amp_raw"].grad is not None
    assert torch.isfinite(amp_params["voltage_amp_raw"].grad)
    assert torch.isfinite(amp_params["current_amp_raw"].grad)
    diagnostics = block.collect_diagnostics()
    assert diagnostics["voltage_amp"] == pytest.approx(1.0)
    assert diagnostics["current_amp"] == pytest.approx(1.0)
