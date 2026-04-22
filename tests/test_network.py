import torch

from digital_drn import SequentialDigitalDRNNet


def test_sequential_network_runs_and_keeps_ff_drive_connected():
    net = SequentialDigitalDRNNet(
        input_dim=3,
        block_configs=[
            {"layer_dims": [4, 2], "weight_gains": [0.1], "bias_gain": 0.0},
            {"layer_dims": [4, 1], "weight_gains": [0.1], "bias_gain": 0.0},
        ],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
    )

    with torch.no_grad():
        for block in net.blocks:
            block.ff[1].weight.fill_(0.5)
            for weight in block.energy.dense_weights:
                weight.state.fill_(0.5)
            for bias in block.energy.biases:
                bias.state.zero_()

    x = torch.randn(5, 3, requires_grad=True)
    out = net(x, reset=True)

    assert out.shape == (5, 1)
    assert net.blocks[0].energy.drive.current.requires_grad

    out.sum().backward()

    assert x.grad is not None
    assert net.blocks[0].ff[1].weight.grad is not None
