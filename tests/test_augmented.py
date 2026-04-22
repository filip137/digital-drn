import torch

from digital_drn import (
    AugmentedFunction,
    ConvDRNBlockEnergy,
    ConvDenseDRNBlockEnergy,
    DenseDRNBlockEnergy,
    Nudging,
    build_dense_drn_block,
)
from digital_drn.core import LinearLayer


def test_nudging_defaults_to_current_and_only_changes_linear_term():
    layer = LinearLayer((3,), batch_size=2)
    layer.state = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]],
        dtype=torch.float32,
    )
    output_gradient = torch.tensor(
        [[0.2, -0.4, 0.6], [-1.0, 2.0, -3.0]],
        dtype=torch.float32,
    )

    nudging = Nudging(layer, nudging=0.5)
    nudging.prepare(output_gradient=output_gradient)

    expected_force = -output_gradient
    expected_b = 0.5 * output_gradient
    expected_energy = -0.5 * torch.sum(layer.state * expected_force, dim=1)

    assert nudging.mode == "current"
    assert torch.allclose(nudging.force, expected_force)
    assert torch.allclose(nudging.eval(), expected_energy)
    assert torch.allclose(nudging.grad_layer_fn(layer)(), expected_b)
    assert nudging.a_coef_fn(layer) is None
    assert torch.allclose(nudging.b_coef_fn(layer)(), expected_b)


def test_augmented_function_defaults_to_current_nudging():
    energy = DenseDRNBlockEnergy(
        layer_dims=[4, 2],
        non_linearity="linear",
        weight_gains=[0.0],
        bias_gain=0.0,
    )
    energy.reset_free_layers(batch_size=2, device=torch.device("cpu"))
    energy.set_drive(torch.zeros(2, 4, dtype=torch.float32))

    augmented = AugmentedFunction(energy)
    output_gradient = torch.ones_like(energy.output_state())
    augmented.prepare_nudging(output_gradient=output_gradient, nudging=0.25)

    assert augmented.nudging_mode == "current"
    assert augmented.nudging.mode == "current"
    assert augmented.layers() == energy.layers()
    assert augmented.params() == energy.params()
    assert torch.allclose(augmented.b_coef_fn(energy.output_layer())(), 0.25 * output_gradient)


def test_energy_helpers_build_current_augmented_energy_and_matching_minimizers():
    dense_energy = DenseDRNBlockEnergy(
        layer_dims=[4, 2],
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )
    conv_energy = ConvDRNBlockEnergy(
        layer_shapes=[(2, 4, 4)],
        kernel_sizes=3,
        non_linearity="linear",
        weight_gains=[],
        bias_gain=0.0,
    )
    mixed_energy = ConvDenseDRNBlockEnergy(
        conv_state_shape=(2, 4, 4),
        output_dim=3,
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    for energy in (dense_energy, conv_energy, mixed_energy):
        augmented = energy.build_augmented_energy()
        inference = energy.build_minimizer(num_iterations=3, mode="asynchronous")
        training = energy.build_minimizer(fn=augmented, num_iterations=3, mode="asynchronous")

        assert augmented.nudging_mode == "current"
        assert augmented.nudging.mode == "current"
        assert inference.num_iterations == 3
        assert inference.mode == "asynchronous"
        assert training.num_iterations == 3
        assert training.mode == "asynchronous"
        assert callable(energy.output_layer)
        assert callable(energy.output_state)


def test_dense_block_builder_exposes_current_augmented_energy_by_default():
    block = build_dense_drn_block(
        input_dim=4,
        layer_dims=[4, 2],
        ff_learning_rate=1.0e-3,
        drn_learning_rate=1.0e-4,
    )

    assert block.augmented_energy.nudging_mode == "current"
    assert block.augmented_energy.nudging.mode == "current"
    assert block.output_layer() is block.energy.output_layer()
    assert block.minimizer is block.inference_minimizer
    assert block.augmented_minimizer is block.training_minimizer
