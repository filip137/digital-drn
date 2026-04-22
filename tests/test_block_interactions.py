import torch

from digital_drn import FFCurrentInteraction
from digital_drn.core import LinearLayer


def test_ff_current_interaction_matches_linear_energy_and_gradient():
    layer = LinearLayer((4,), batch_size=2)
    layer.state = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.5, -1.5]],
        dtype=torch.float32,
    )
    current = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [-1.0, 1.0, -2.0, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    interaction = FFCurrentInteraction(layer)
    interaction.set_current(current)

    expected_energy = -torch.sum(layer.state * current, dim=1)

    assert torch.allclose(interaction.eval(), expected_energy)
    assert torch.allclose(interaction.grad_layer_fn(layer)(), -current)
    assert interaction.params() == []
