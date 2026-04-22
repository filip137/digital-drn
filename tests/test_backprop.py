import torch
from torch import nn

from digital_drn import SequentialDigitalDRNNet


def test_backprop_helpers_expose_and_update_resistive_states():
    net = SequentialDigitalDRNNet(
        input_dim=3,
        block_configs=[{"layer_dims": [4, 2], "weight_gains": [0.1], "bias_gain": 0.0}],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        weight_min=0.0,
        weight_max=1.0,
    )
    net.set_device(torch.device("cpu"))
    net.enable_resistive_grad_()

    optimizer = torch.optim.SGD(net.optimizer_tensors(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))

    initial_weight = net.blocks[0].energy.dense_weights[0].state.detach().clone()
    initial_drive_scale = net.blocks[0].drive_scale.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = net(x, reset=True)
    loss = criterion(logits, y)
    loss.backward()

    assert net.blocks[0].ff[1].weight.grad is not None
    assert net.blocks[0]._drive_scale_raw.grad is not None
    assert net.blocks[0].energy.dense_weights[0].state.grad is not None

    optimizer.step()
    net.clamp_resistive_params_()
    net.detach_state_()

    updated_weight = net.blocks[0].energy.dense_weights[0].state.detach()
    updated_drive_scale = net.blocks[0].drive_scale.detach()
    assert not torch.allclose(initial_weight, updated_weight)
    assert not torch.allclose(initial_drive_scale, updated_drive_scale)
