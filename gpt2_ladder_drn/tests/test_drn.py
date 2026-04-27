import torch

from gpt2_ladder_drn import DRNCell, DebugGPT2Config, GPT2LMHeadModel, LadderSideGPT2


def test_drn_cell_backpropagates_through_unrolled_iterations():
    torch.manual_seed(4)
    drn = DRNCell(d_model=32, n_iter=4)
    x = torch.randn(2, 16, 32, requires_grad=True)

    y = drn(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert all(param.grad is not None for param in drn.parameters() if param.requires_grad)


def test_ladder_drn_block_runs_with_debug_gpt2():
    torch.manual_seed(5)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        side_block_type="drn",
        drn_iter=2,
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 10))

    out = model(input_ids, targets=input_ids)
    out["loss"].backward()

    assert out["logits"].shape == (2, 10, cfg.vocab_size)
    assert all(param.grad is None for param in model.base.parameters())

    drn_mlp = model.side_blocks[0].drn
    assert drn_mlp.block.inference_minimizer.mode == "asynchronous"
    assert drn_mlp.block.energy._non_linearity == "perfect_diode"
    assert drn_mlp.block.energy.drive.current.shape[0] == input_ids.numel()

    resistive_grads = [tensor.grad for _name, tensor in model.named_resistive_parameters()]
    assert resistive_grads
    assert all(grad is not None for grad in resistive_grads)
