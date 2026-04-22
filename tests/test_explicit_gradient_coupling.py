import torch
import torch.nn as nn

from digital_drn import BlockEquilibriumProp, build_dense_drn_block


def test_last_head_bp_and_last_drn_ep_centered_current_nudging():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=4,
        mode="asynchronous",
        non_linearity="linear",
        ff_learning_rate=1.0e-3,
        drn_learning_rate=1.0e-4,
        weight_gains=[0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    head = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        block.ff[1].weight.fill_(0.1)
        block.ff[1].bias.zero_()
        for weight in block.energy.dense_weights:
            weight.state.fill_(0.1)
        for bias in block.energy.biases:
            bias.state.zero_()
        head.weight.copy_(
            torch.tensor(
                [
                    [0.3, -0.2],
                    [-0.1, 0.4],
                    [0.2, 0.1],
                ],
                dtype=torch.float32,
            )
        )
        head.bias.zero_()

    inputs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
            [-0.3, 0.2, 0.1],
            [0.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    h_free = block(inputs, reset=True, num_iterations=4)
    assert torch.isfinite(h_free).all()
    free_cache = block.capture_free_cache(inputs)

    # Step 1: the last digital readout gets direct BP gradients only.
    h_leaf = h_free.detach().clone().requires_grad_(True)
    logits = head(h_leaf)
    loss = criterion(logits, targets)
    delta_h, *head_grads = torch.autograd.grad(loss, (h_leaf, *head.parameters()))

    assert delta_h.shape == h_free.shape
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in head_grads)

    # Step 2: the last DRN block is nudged by the fixed free-phase output gradient.
    beta = 0.1
    ep = BlockEquilibriumProp(block, beta=beta)
    result = ep.compute_gradients(
        free_cache=free_cache,
        output_cotangent=delta_h,
    )

    assert torch.allclose(block.augmented_energy.nudging.force, -delta_h.detach())
    assert torch.max(torch.abs(result.plus_output - result.minus_output)).item() > 0.0

    assert len(result.param_grads) == len(block.resistive_params())
    assert all(
        grad.shape == param.state.shape
        for grad, param in zip(result.param_grads, block.resistive_params())
    )
    assert result.delta_drive.shape == free_cache.drive.shape
    assert any(torch.count_nonzero(grad).item() > 0 for grad in result.param_grads)
