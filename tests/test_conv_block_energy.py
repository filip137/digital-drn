import torch

from digital_drn import ConvDRNBlockEnergy, ConvLayer, ConvResistive, ConvWeight, QuadraticMinimizer
from digital_drn.core import SumSeparableFunction


def test_conv_block_energy_includes_drive_in_b_coefficient_and_resets_state():
    energy = ConvDRNBlockEnergy(
        layer_shapes=[(4, 8, 8), (4, 8, 8)],
        kernel_sizes=[3],
        paddings=[1],
        strides=[1],
        non_linearity="linear",
        weight_gains=[0.0],
        bias_gain=0.0,
    )
    energy.reset_free_layers(batch_size=2, device=torch.device("cpu"))

    current = torch.randn(2, 4, 8, 8, requires_grad=True)
    energy.set_drive(current)

    first_layer = energy.free_layers()[0]
    b_coef = energy.b_coef_fn(first_layer)()

    assert torch.allclose(b_coef, -current)
    assert energy.free_layers() == energy.layers()
    for layer in energy.free_layers():
        assert torch.allclose(layer.state, torch.zeros_like(layer.state))


def test_single_conv_quadratic_update_reduces_output_gradient():
    torch.manual_seed(0)

    input_layer = ConvLayer((4, 6, 6), device="cpu", non_linearity="linear")
    output_layer = ConvLayer((4, 6, 6), device="cpu", non_linearity="linear")
    conv_weight = ConvWeight(
        shape=(4, 4, 3, 3),
        gain=0.2,
        device="cpu",
        clamp=False,
        clamp_min=None,
        clamp_max=None,
        init_mode="kaiming_uniform",
    )
    interaction = ConvResistive(
        input_layer,
        output_layer,
        conv_weight,
        padding=1,
        stride=1,
        dilation=1,
        voltage_amp=1.0,
        current_amp=1.0,
    )
    energy_fn = SumSeparableFunction([input_layer, output_layer], [conv_weight], [interaction])

    input_layer.state = torch.randn(2, 4, 6, 6)
    output_layer.state = torch.zeros(2, 4, 6, 6)

    minimizer = QuadraticMinimizer(
        energy_fn,
        free_layers=[output_layer],
        num_iterations=3,
        mode="forward",
        non_linearity="linear",
        quadratic_diode_param={},
        exponential_diode_param={},
        voltage_amp=1.0,
        current_amp=1.0,
        hard_sigmoid_param={},
    )

    grad_before = energy_fn.grad_layer_fn(output_layer)().norm().item()
    minimizer.compute_equilibrium()
    grad_after = energy_fn.grad_layer_fn(output_layer)().norm().item()

    assert grad_after < grad_before * 1.0e-3


def test_conv_block_can_make_last_state_linear_while_hidden_states_remain_nonlinear():
    energy = ConvDRNBlockEnergy(
        layer_shapes=[(4, 6, 6), (4, 6, 6), (4, 6, 6)],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1],
        non_linearity="perfect_diode",
        output_non_linearity="linear",
        weight_gains=[0.0, 0.0],
        bias_gain=0.0,
    )

    hidden_0, hidden_1, output = energy.free_layers()
    assert hidden_0.non_linearity == "perfect_diode"
    assert hidden_1.non_linearity == "perfect_diode"
    assert output.non_linearity == "linear"

    hidden_0.state = torch.tensor([[[[-1.0]], [[2.0]], [[3.0]], [[-4.0]]]], dtype=torch.float32)
    output.state = torch.tensor([[[[-1.0]], [[2.0]], [[3.0]], [[-4.0]]]], dtype=torch.float32)

    hidden_act = hidden_0.activate()
    output_act = output.activate()

    assert torch.allclose(hidden_act, torch.tensor([[[[0.0]], [[2.0]], [[0.0]], [[-4.0]]]], dtype=torch.float32))
    assert torch.allclose(output_act, output.state)


def test_conv_resistive_can_skip_amplification_on_current_injected_layer1():
    input_layer = ConvLayer((4, 4, 4), device="cpu", non_linearity="linear")
    output_layer = ConvLayer((4, 4, 4), device="cpu", non_linearity="linear")
    input_layer._name = "Layer_1"
    output_layer._name = "Layer_2"
    conv_weight = ConvWeight(
        shape=(4, 4, 3, 3),
        gain=0.2,
        device="cpu",
        clamp=False,
        clamp_min=None,
        clamp_max=None,
        init_mode="kaiming_uniform",
    )
    input_layer.state = torch.randn(2, 4, 4, 4)

    interaction_default = ConvResistive(
        input_layer,
        output_layer,
        conv_weight,
        padding=1,
        stride=1,
        dilation=1,
        voltage_amp=4.0,
        current_amp=1.0,
        amplify_first_free_layer=True,
    )
    interaction_skip = ConvResistive(
        input_layer,
        output_layer,
        conv_weight,
        padding=1,
        stride=1,
        dilation=1,
        voltage_amp=4.0,
        current_amp=1.0,
        amplify_first_free_layer=False,
    )

    default_b_post = interaction_default.b_coef_fn(output_layer)()
    skip_b_post = interaction_skip.b_coef_fn(output_layer)()

    assert torch.allclose(default_b_post, skip_b_post * 4.0, atol=1.0e-6, rtol=1.0e-6)
