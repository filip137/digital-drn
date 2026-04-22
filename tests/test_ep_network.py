import torch
import torch.nn as nn

from digital_drn import DigitalAnalogNet, build_dense_drn_block, hybrid_backward_explicit


def _configure_stable_dense_block(block):
    with torch.no_grad():
        block.ff[1].weight.fill_(0.1)
        block.ff[1].bias.zero_()
        for weight in block.energy.dense_weights:
            weight.state.fill_(0.1)
        for bias in block.energy.biases:
            bias.state.zero_()


def _flat(tensors):
    return torch.cat([tensor.reshape(-1).detach().float() for tensor in tensors])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item() / (a.norm().item() * b.norm().item()))


def test_hybrid_backward_explicit_matches_bp_on_head_and_digital_coupling():
    torch.manual_seed(0)

    block1 = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=8,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    block2 = build_dense_drn_block(
        input_dim=2,
        layer_dims=[4, 2],
        num_iterations=8,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block1)
    _configure_stable_dense_block(block2)

    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float32))
        head.bias.zero_()

    model = DigitalAnalogNet([block1, block2], head=head).set_device(torch.device("cpu"))
    model.enable_resistive_grad_()
    criterion = nn.CrossEntropyLoss()

    inputs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
            [-0.3, 0.2, 0.1],
            [0.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    for block in model.blocks:
        for resistive_param in block.resistive_params():
            resistive_param.state.grad = None

    logits = model(inputs, reset=True, num_iterations=8)
    loss = criterion(logits, targets)
    loss.backward()

    bp_head = [param.grad.detach().clone() for param in model.head.parameters()]
    bp_blocks = []
    for block in model.blocks:
        bp_blocks.append(
            {
                "ff": [param.grad.detach().clone() for param in block.ff.parameters()],
                "drive": block._drive_scale_raw.grad.detach().clone(),
            }
        )
    model.detach_state_()

    hybrid = hybrid_backward_explicit(
        model,
        inputs,
        targets,
        criterion=criterion,
        beta=1.0e-3,
        reset=True,
        num_iterations=8,
    )

    head_cos = _cosine(_flat(bp_head), _flat(hybrid.head.head_param_grads))
    assert head_cos > 0.99999

    for bp_block, hybrid_block in zip(bp_blocks, hybrid.blocks):
        ff_cos = _cosine(_flat(bp_block["ff"]), _flat(hybrid_block.digital.ff_param_grads))
        drive_cos = _cosine(
            bp_block["drive"].reshape(-1).float(),
            hybrid_block.digital.drive_scale_grad.reshape(-1).float(),
        )
        assert ff_cos > 0.999
        assert drive_cos > 0.999


def test_hybrid_backward_explicit_matches_bp_on_head_and_digital_coupling_with_signed_drive():
    torch.manual_seed(0)

    block1 = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        signed_drive=True,
        ff_activation="identity",
        num_iterations=8,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    block2 = build_dense_drn_block(
        input_dim=2,
        layer_dims=[4, 2],
        signed_drive=True,
        ff_activation="identity",
        num_iterations=8,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    with torch.no_grad():
        ff1 = list(block1.ff.parameters())
        ff2 = list(block2.ff.parameters())
        ff1[0].copy_(torch.tensor([[0.3, -0.1, 0.2], [-0.2, 0.4, 0.1]], dtype=torch.float32))
        ff1[1].copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        ff2[0].copy_(torch.tensor([[0.25, -0.35], [0.15, 0.05]], dtype=torch.float32))
        ff2[1].copy_(torch.tensor([0.05, -0.1], dtype=torch.float32))
        for block, weight_tensor in (
            (
                block1,
                torch.tensor(
                    [[0.11, 0.07], [0.09, 0.13], [0.05, 0.12], [0.14, 0.08]],
                    dtype=torch.float32,
                ),
            ),
            (
                block2,
                torch.tensor(
                    [[0.08, 0.10], [0.12, 0.06], [0.07, 0.11], [0.09, 0.13]],
                    dtype=torch.float32,
                ),
            ),
        ):
            for weight in block.energy.dense_weights:
                weight.state.copy_(weight_tensor)
            for bias in block.energy.biases:
                bias.state.copy_(torch.tensor([0.01, -0.02, 0.03, -0.01], dtype=torch.float32))
        block1._drive_scale_raw.fill_(0.4)
        block2._drive_scale_raw.fill_(0.5)

    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.3, -0.2], [-0.1, 0.4]], dtype=torch.float32))
        head.bias.copy_(torch.tensor([0.05, -0.03], dtype=torch.float32))

    model = DigitalAnalogNet([block1, block2], head=head).set_device(torch.device("cpu"))
    model.enable_resistive_grad_()
    criterion = nn.CrossEntropyLoss()

    inputs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
            [-0.3, 0.2, 0.1],
            [0.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    for block in model.blocks:
        for resistive_param in block.resistive_params():
            resistive_param.state.grad = None

    logits = model(inputs, reset=True, num_iterations=8)
    loss = criterion(logits, targets)
    loss.backward()

    bp_head = [param.grad.detach().clone() for param in model.head.parameters()]
    bp_blocks = []
    for block in model.blocks:
        bp_blocks.append(
            {
                "ff": [param.grad.detach().clone() for param in block.ff.parameters()],
                "drive": block._drive_scale_raw.grad.detach().clone(),
            }
        )
    model.detach_state_()

    hybrid = hybrid_backward_explicit(
        model,
        inputs,
        targets,
        criterion=criterion,
        beta=1.0e-3,
        reset=True,
        num_iterations=8,
    )

    head_cos = _cosine(_flat(bp_head), _flat(hybrid.head.head_param_grads))
    assert head_cos > 0.99999

    for bp_block, hybrid_block in zip(bp_blocks, hybrid.blocks):
        ff_cos = _cosine(_flat(bp_block["ff"]), _flat(hybrid_block.digital.ff_param_grads))
        drive_cos = _cosine(
            bp_block["drive"].reshape(-1).float(),
            hybrid_block.digital.drive_scale_grad.reshape(-1).float(),
        )
        assert ff_cos > 0.999
        assert drive_cos > 0.999
