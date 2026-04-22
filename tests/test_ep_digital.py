import torch

from digital_drn import build_dense_drn_block, vjp_ff_block


def test_vjp_ff_block_matches_direct_autograd():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        ff_activation="identity",
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    with torch.no_grad():
        block.ff[1].weight.copy_(
            torch.tensor(
                [
                    [0.3, -0.1, 0.2],
                    [-0.2, 0.4, 0.1],
                    [0.1, 0.0, -0.3],
                    [0.2, 0.2, 0.2],
                ],
                dtype=torch.float32,
            )
        )
        block.ff[1].bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3], dtype=torch.float32))
        block._drive_scale_raw.fill_(0.4)

    h_prev = torch.tensor(
        [
            [0.5, -1.0, 0.2],
            [0.1, 0.3, -0.4],
        ],
        dtype=torch.float32,
    )
    delta_drive = torch.tensor(
        [
            [0.2, -0.3, 0.1, 0.5],
            [-0.4, 0.6, -0.2, 0.1],
        ],
        dtype=torch.float32,
    )

    result = vjp_ff_block(block, h_prev, delta_drive)

    h_prev_leaf = h_prev.detach().clone().requires_grad_(True)
    ff_params = list(block.ff.parameters())
    direct_grads = torch.autograd.grad(
        outputs=block.drive_scale * block.ff(h_prev_leaf),
        inputs=(h_prev_leaf, *ff_params, block._drive_scale_raw),
        grad_outputs=delta_drive,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert len(result.ff_params) == len(ff_params)
    assert all(param is expected for param, expected in zip(result.ff_params, ff_params))
    assert torch.allclose(result.delta_h_prev, direct_grads[0])
    assert all(
        torch.allclose(grad, direct)
        for grad, direct in zip(result.ff_param_grads, direct_grads[1 : 1 + len(ff_params)])
    )
    assert result.drive_scale_grad is not None
    assert torch.allclose(result.drive_scale_grad, direct_grads[-1])


def test_vjp_ff_block_matches_signed_drive_difference_rule():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        ff_activation="identity",
        signed_drive=True,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    base_linear = block.ff.base_ff[1]
    with torch.no_grad():
        base_linear.weight.copy_(
            torch.tensor(
                [
                    [0.3, -0.1, 0.2],
                    [-0.2, 0.4, 0.1],
                ],
                dtype=torch.float32,
            )
        )
        base_linear.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        block._drive_scale_raw.fill_(0.4)

    h_prev = torch.tensor(
        [
            [0.5, -1.0, 0.2],
            [0.1, 0.3, -0.4],
        ],
        dtype=torch.float32,
    )
    delta_drive = torch.tensor(
        [
            [0.2, -0.3, 0.1, 0.5],
            [-0.4, 0.6, -0.2, 0.1],
        ],
        dtype=torch.float32,
    )

    result = vjp_ff_block(block, h_prev, delta_drive)

    h_prev_leaf = h_prev.detach().clone().requires_grad_(True)
    base_output = block.ff.base_ff(h_prev_leaf)
    reduced_cotangent = block.drive_scale.detach() * (delta_drive[:, :2] - delta_drive[:, 2:])
    direct_grads = torch.autograd.grad(
        outputs=base_output,
        inputs=(h_prev_leaf, *block.ff.base_ff.parameters()),
        grad_outputs=reduced_cotangent,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert torch.allclose(result.delta_h_prev, direct_grads[0])
    assert all(
        torch.allclose(grad, direct)
        for grad, direct in zip(result.ff_param_grads, direct_grads[1:])
    )
