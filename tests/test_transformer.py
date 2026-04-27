import pytest
import torch
import torch.nn.functional as F

from digital_drn import DRNGPTConfig, ResistiveTrainableModel, SmallDRNGPT


def _tiny_config(**overrides):
    values = {
        "vocab_size": 64,
        "seq_len": 16,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "drn_num_iterations": 2,
        "drn_non_linearity": "perfect_diode",
        "drn_weight_gains": 0.1,
        "drn_bias_gain": 0.0,
    }
    values.update(overrides)
    return DRNGPTConfig(**values)


def test_small_drn_gpt_shape_and_default_perfect_diode():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = SmallDRNGPT(cfg).set_device(torch.device("cpu"))
    model.enable_resistive_grad_()

    input_ids = torch.randint(0, cfg.vocab_size, (3, cfg.seq_len))
    logits = model(input_ids)

    assert isinstance(model, ResistiveTrainableModel)
    assert logits.shape == (3, cfg.seq_len, cfg.vocab_size)
    assert model.blocks[0].mlp.block.energy._non_linearity == "perfect_diode"
    assert model.blocks[0].mlp.block.minimizer.mode == "asynchronous"


def test_small_drn_gpt_causal_mask_prevents_future_leakage():
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = SmallDRNGPT(cfg).set_device(torch.device("cpu")).eval()

    x1 = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    x2 = x1.clone()
    split = 7
    x2[:, split + 1 :] = torch.randint(0, cfg.vocab_size, x2[:, split + 1 :].shape)

    with torch.no_grad():
        logits1 = model(x1, reset=True)
        logits2 = model(x2, reset=True)

    assert torch.allclose(logits1[:, : split + 1], logits2[:, : split + 1], atol=1.0e-5, rtol=1.0e-5)


def test_small_drn_gpt_backprop_updates_resistive_state():
    torch.manual_seed(2)
    cfg = _tiny_config(n_layers=1)
    model = SmallDRNGPT(cfg).set_device(torch.device("cpu"))
    model.enable_resistive_grad_()
    optimizer = torch.optim.SGD(model.optimizer_tensors(), lr=0.1)

    input_ids = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    initial_weight = model.blocks[0].mlp.block.energy.dense_weights[0].state.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids, reset=True)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    assert model.blocks[0].mlp.block.energy.dense_weights[0].state.grad is not None
    assert any(tensor is model.blocks[0].mlp.block.energy.dense_weights[0].state for tensor in model.optimizer_tensors())

    optimizer.step()
    model.clamp_resistive_params_()
    model.detach_state_()

    updated_weight = model.blocks[0].mlp.block.energy.dense_weights[0].state.detach()
    assert not torch.allclose(initial_weight, updated_weight)


def test_small_drn_gpt_rejects_odd_nonlinear_hidden_width():
    with pytest.raises(ValueError, match="DRN hidden dimension must be even"):
        SmallDRNGPT(_tiny_config(drn_hidden_dim=15))
