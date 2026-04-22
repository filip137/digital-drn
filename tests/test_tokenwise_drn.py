import torch

from digital_drn import TokenwiseDRNMLP


def test_tokenwise_drn_mlp_preserves_transformer_shape_and_grads():
    torch.manual_seed(0)
    module = TokenwiseDRNMLP(
        d_model=6,
        hidden_dim=8,
        dropout=0.0,
        num_iterations=3,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    module.enable_resistive_grad_()

    x = torch.randn(2, 4, 6)
    y = module(x)

    assert y.shape == x.shape

    loss = y.square().mean()
    loss.backward()

    assert module.block.ff[1].weight.grad is not None
    assert module.block._drive_scale_raw.grad is not None
    assert module.block.energy.dense_weights[0].state.grad is not None


def test_tokenwise_drn_mlp_does_not_mix_tokens_across_flattened_axis():
    torch.manual_seed(1)
    module = TokenwiseDRNMLP(
        d_model=4,
        hidden_dim=6,
        dropout=0.0,
        num_iterations=4,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    module.eval()

    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        batched = module(x, reset=True)
        per_token = torch.empty_like(batched)
        for batch_idx in range(x.size(0)):
            for token_idx in range(x.size(1)):
                per_token[batch_idx, token_idx] = module(
                    x[batch_idx : batch_idx + 1, token_idx : token_idx + 1],
                    reset=True,
                )[0, 0]

    assert torch.allclose(batched, per_token, atol=1.0e-6, rtol=1.0e-5)


def test_tokenwise_drn_mlp_exposes_resistive_states_for_optimizers():
    module = TokenwiseDRNMLP(
        d_model=4,
        hidden_dim=6,
        dropout=0.0,
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    )

    optimizer_tensors = module.optimizer_tensors()

    assert any(tensor is module.block.energy.dense_weights[0].state for tensor in optimizer_tensors)
