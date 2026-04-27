import torch

from gpt2_ladder_drn import DebugGPT2Config, GPT2LMHeadModel, LadderSideGPT2


def test_ladder_side_gpt2_shapes_and_side_gradients_only():
    torch.manual_seed(2)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(base, reduction_factor=8, num_side_layers=2)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    out = model(input_ids, targets=input_ids)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)

    out["loss"].backward()

    assert all(param.grad is None for param in model.base.parameters())
    side_grads = [
        param.grad
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("base.")
    ]
    assert side_grads
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in side_grads)


def test_ladder_residual_logits_mode():
    torch.manual_seed(3)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        output_mode="residual_logits",
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    out = model(input_ids, targets=input_ids)

    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert model.gamma is not None
