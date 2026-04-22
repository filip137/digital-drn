import torch

from digital_drn import DenseDRNBlockEnergy


def test_block_energy_includes_drive_in_b_coefficient_and_resets_state():
    energy = DenseDRNBlockEnergy(
        layer_dims=[4, 2],
        non_linearity="linear",
        weight_gains=[0.0],
        bias_gain=0.0,
    )
    energy.reset_free_layers(batch_size=3, device=torch.device("cpu"))

    current = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5],
            [0.1, 0.2, 0.3, 0.4],
            [-0.7, 0.8, -0.9, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    energy.set_drive(current)

    first_layer = energy.free_layers()[0]
    b_coef = energy.b_coef_fn(first_layer)()

    assert torch.allclose(b_coef, -current)
    assert energy.free_layers() == energy.layers()
    for layer in energy.free_layers():
        assert torch.allclose(layer.state, torch.zeros_like(layer.state))
