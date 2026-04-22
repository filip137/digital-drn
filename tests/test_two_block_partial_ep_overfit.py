import torch
import torch.nn as nn

from digital_drn import BlockEquilibriumProp, build_dense_drn_block


def _configure_prefix_stable(block):
    with torch.no_grad():
        block.ff[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        block.ff[1].bias.zero_()
        for weight in block.energy.dense_weights:
            weight.state.fill_(0.5)
        for bias in block.energy.biases:
            bias.state.zero_()


def _configure_last_block_asymmetric(block):
    with torch.no_grad():
        block.ff[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        block.ff[1].bias.zero_()
        block.energy.dense_weights[0].state.copy_(
            torch.tensor(
                [
                    [0.30, 0.10],
                    [0.10, 0.30],
                    [0.20, 0.05],
                    [0.05, 0.20],
                ],
                dtype=torch.float32,
            )
        )
        for bias in block.energy.biases:
            bias.state.zero_()


def test_two_block_network_overfits_with_fixed_head_and_last_block_ep():
    torch.manual_seed(0)

    block1 = build_dense_drn_block(
        input_dim=2,
        layer_dims=[4, 2],
        num_iterations=3,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
        ff_learning_rate=None,
        drn_learning_rate=None,
        weight_min=1.0e-6,
        weight_max=2.0,
    ).set_device(torch.device("cpu"))
    block2 = build_dense_drn_block(
        input_dim=2,
        layer_dims=[4, 2],
        num_iterations=3,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
        ff_learning_rate=None,
        drn_learning_rate=0.2,
        weight_min=1.0e-6,
        weight_max=2.0,
    ).set_device(torch.device("cpu"))

    # Keep the readout fixed so the DRN path is responsible for fitting.
    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        head.bias.zero_()
    for param in head.parameters():
        param.requires_grad_(False)

    _configure_prefix_stable(block1)
    _configure_last_block_asymmetric(block2)

    inputs = torch.tensor(
        [
            [1.0, 1.0],
            [1.2, 0.8],
            [0.8, 1.1],
            [1.1, 1.3],
            [-1.0, -1.1],
            [-1.2, -0.8],
            [-0.9, -1.3],
            [-1.1, -0.9],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(block2.resistive_param_states(), lr=0.2)
    beta = 0.1

    initial_last_drn = [param.state.detach().clone() for param in block2.resistive_params()]

    with torch.no_grad():
        h1_init = block1(inputs, reset=True, num_iterations=3)
        h2_init = block2(h1_init, reset=True, num_iterations=3)
        initial_preds = head(h2_init).argmax(dim=1)
        initial_acc = float((initial_preds == targets).float().mean().item())
        block1.detach_state_()
        block2.detach_state_()

    final_acc = initial_acc
    for _epoch in range(40):
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            h1 = block1(inputs, reset=True, num_iterations=3)

        h2_free = block2(h1.detach(), reset=True, num_iterations=3)
        free_cache = block2.capture_free_cache(h1.detach())

        h2_leaf = h2_free.detach().clone().requires_grad_(True)
        logits = head(h2_leaf)
        loss = criterion(logits, targets)
        (delta_h,) = torch.autograd.grad(loss, (h2_leaf,))

        ep = BlockEquilibriumProp(block2, beta=beta)
        result = ep.compute_gradients(
            free_cache=free_cache,
            output_cotangent=delta_h,
        )
        for param, grad in zip(block2.resistive_params(), result.param_grads):
            param.state.grad = grad.detach().clone()

        optimizer.step()
        block2.clamp_resistive_params_()
        block1.detach_state_()
        block2.detach_state_()

        with torch.no_grad():
            h1_eval = block1(inputs, reset=True, num_iterations=3)
            h2_eval = block2(h1_eval, reset=True, num_iterations=3)
            preds = head(h2_eval).argmax(dim=1)
            final_acc = float((preds == targets).float().mean().item())
            block1.detach_state_()
            block2.detach_state_()
        if final_acc == 1.0:
            break

    param_changes = {
        getattr(param, "name", param.__class__.__name__): float(
            (param.state.detach() - init).abs().max().item()
        )
        for param, init in zip(block2.resistive_params(), initial_last_drn)
    }
    max_param_change = max(param_changes.values())
    dense_weight_change = max(
        change for name, change in param_changes.items() if "DenseWeight" in name
    )

    assert initial_acc == 0.5
    assert final_acc >= 0.99
    assert max_param_change > 1.0e-4
    assert dense_weight_change > 1.0e-4
