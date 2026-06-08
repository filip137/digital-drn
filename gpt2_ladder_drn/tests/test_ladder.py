import pytest
import torch

from gpt2_ladder_drn import DebugGPT2Config, GPT2LMHeadModel, LadderSideGPT2, structural_init_from_backbone


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


def test_ladder_gated_logits_matches_upstream_logit_mixture():
    torch.manual_seed(31)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        output_mode="gated_logits",
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    out = model(input_ids, targets=input_ids, return_hidden_states=True)

    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert model.gamma is None
    assert model.logit_gate is not None
    rho = torch.sigmoid(model.logit_gate / model.temperature)
    torch.testing.assert_close(rho, torch.tensor(0.5))

    with torch.no_grad():
        base_out = base(input_ids, targets=None, return_hidden_states=True)
        final_side_state = out["hidden_states"][-1]
        side_logits = model.base.lm_head(model.up_proj(model.side_ln_f(final_side_state)))
        expected_logits = rho * side_logits + (1.0 - rho) * base_out["logits"]
    torch.testing.assert_close(out["logits"], expected_logits)

    diagnostics = model.collect_ladder_diagnostics()
    assert diagnostics["logit_gate_alpha"] == pytest.approx(0.0)
    assert diagnostics["logit_gate_rho"] == pytest.approx(0.5)


def test_digital_post_residual_ladder_mode_shapes_and_gradients():
    torch.manual_seed(32)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        output_mode="gated_logits",
        side_block_type="digital",
        ladder_injection_mode="post_drn_residual",
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    out = model(input_ids, targets=input_ids, return_hidden_states=True)

    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert len(out["hidden_states"]) == 3
    assert model.post_drn_alphas is None
    assert model.post_ladder_lambdas is not None
    assert model.logit_gate is not None

    out["loss"].backward()

    assert all(param.grad is None for param in model.base.parameters())
    assert model.post_ladder_lambdas.grad is not None
    assert torch.count_nonzero(model.post_ladder_lambdas.grad).item() > 0

    diagnostics = model.collect_ladder_diagnostics()
    assert diagnostics["ladder_injection_mode"] == "post_drn_residual"
    assert diagnostics["mean_abs_lambda"] > 0.0
    assert "mean_abs_alpha" not in diagnostics
    assert diagnostics["z_side_norm"] > 0.0


def test_ladder_accepts_explicit_tap_indices_and_full_initial_tap():
    torch.manual_seed(4)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        tap_indices=[1, 2],
        initial_side_state_mode="full_tap",
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    out = model(input_ids, targets=input_ids, return_hidden_states=True)

    assert model.base_indices == [1, 2]
    assert len(out["hidden_states"]) == 3
    with torch.no_grad():
        base_out = base(input_ids, return_hidden_states=True)
        expected_initial = model.down_projs[0](base_out["hidden_states"][0].detach())
    torch.testing.assert_close(out["hidden_states"][0], expected_initial)


def test_ladder_rejects_unsorted_tap_indices():
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    with pytest.raises(ValueError, match="ascending"):
        LadderSideGPT2(base, reduction_factor=8, tap_indices=[2, 1])


def test_magnitude_pruned_structural_init_copies_digital_side_block_weights():
    torch.manual_seed(5)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(base, reduction_factor=8, tap_indices=[1, 2])

    structural_init_from_backbone(model, base, method="magnitude_pruned")

    cols = torch.topk(base.transformer.wte.weight.detach().abs().sum(dim=0), k=model.d_side).indices.sort().values
    qkv_cols = torch.cat([cols + offset * model.d_model for offset in range(3)])
    mlp_cols = torch.cat([cols + offset * model.d_model for offset in range(4)])

    torch.testing.assert_close(model.down_projs[0].weight[:, cols], torch.eye(model.d_side))
    torch.testing.assert_close(model.up_proj.weight[cols, :], torch.eye(model.d_side))
    torch.testing.assert_close(
        model.side_blocks[0].ln_1.weight,
        base.transformer.h[0].ln_1.weight.detach().index_select(0, cols),
    )
    torch.testing.assert_close(
        model.side_blocks[0].attn.c_attn.weight,
        base.transformer.h[0].attn.c_attn.weight.detach().index_select(0, qkv_cols).index_select(1, cols),
    )
    torch.testing.assert_close(
        model.side_blocks[0].mlp.c_fc.weight,
        base.transformer.h[0].mlp.c_fc.weight.detach().index_select(0, mlp_cols).index_select(1, cols),
    )


def test_magnitude_attn_structural_init_copies_drn_hybrid_attention_weights():
    torch.manual_seed(51)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        tap_indices=[1, 2],
        side_block_type="drn_hybrid_attn",
        drn_iter=2,
    )

    structural_init_from_backbone(model, base, method="magnitude_attn")

    cols = torch.topk(base.transformer.wte.weight.detach().abs().sum(dim=0), k=model.d_side).indices.sort().values
    qkv_cols = torch.cat([cols + offset * model.d_model for offset in range(3)])

    torch.testing.assert_close(model.down_projs[0].weight[:, cols], torch.eye(model.d_side))
    torch.testing.assert_close(model.up_proj.weight[cols, :], torch.eye(model.d_side))
    torch.testing.assert_close(
        model.side_blocks[0].ln_1.weight,
        base.transformer.h[0].ln_1.weight.detach().index_select(0, cols),
    )
    torch.testing.assert_close(
        model.side_blocks[0].ln_2.weight,
        base.transformer.h[0].ln_2.weight.detach().index_select(0, cols),
    )
    torch.testing.assert_close(
        model.side_blocks[0].attn.c_attn.weight,
        base.transformer.h[0].attn.c_attn.weight.detach().index_select(0, qkv_cols).index_select(1, cols),
    )
    torch.testing.assert_close(
        model.side_blocks[0].attn.c_proj.weight,
        base.transformer.h[0].attn.c_proj.weight.detach().index_select(0, cols).index_select(1, cols),
    )


def test_post_drn_residual_mode_shapes_freezing_and_gradients():
    torch.manual_seed(6)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        output_mode="residual_logits",
        side_block_type="drn_hybrid_attn",
        ladder_injection_mode="post_drn_residual",
        drn_iter=2,
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 10))

    out = model(input_ids, targets=input_ids, return_hidden_states=True)
    assert out["logits"].shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert len(out["hidden_states"]) == 3
    assert model.gamma is not None
    assert model.gamma.detach().abs().item() > 0.0

    out["loss"].backward()

    assert all(param.grad is None for param in model.base.parameters())
    assert model.post_drn_alphas.grad is not None
    assert model.post_ladder_lambdas.grad is not None
    assert torch.count_nonzero(model.post_drn_alphas.grad).item() > 0
    assert torch.count_nonzero(model.post_ladder_lambdas.grad).item() > 0

    resistive_grads = [tensor.grad for _name, tensor in model.named_resistive_parameters()]
    assert resistive_grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in resistive_grads)

    diagnostics = model.collect_ladder_diagnostics()
    assert diagnostics["ladder_injection_mode"] == "post_drn_residual"
    assert diagnostics["post_drn_feedback"] is True
    assert diagnostics["mean_abs_alpha"] > 0.0
    assert diagnostics["mean_abs_lambda"] > 0.0
    assert diagnostics["z_side_norm"] > 0.0


def test_drn_ladder_uses_signed_drive_by_default_and_can_disable_it():
    torch.manual_seed(61)
    cfg = DebugGPT2Config(dropout=0.0)
    signed_model = LadderSideGPT2(
        GPT2LMHeadModel(cfg),
        reduction_factor=8,
        num_side_layers=2,
        side_block_type="drn_hybrid_attn",
        drn_iter=2,
    )
    unsigned_model = LadderSideGPT2(
        GPT2LMHeadModel(cfg),
        reduction_factor=8,
        num_side_layers=2,
        side_block_type="drn_pure",
        drn_iter=2,
        drn_signed_drive=False,
    )

    assert signed_model.drn_signed_drive is True
    assert signed_model.drn_mlps()
    assert all(mlp.block.signed_drive is True for mlp in signed_model.drn_mlps())
    assert signed_model.collect_ladder_diagnostics()["drn_signed_drive"] is True

    assert unsigned_model.drn_signed_drive is False
    assert unsigned_model.drn_mlps()
    assert all(mlp.block.signed_drive is False for mlp in unsigned_model.drn_mlps())


def test_drn_ladder_accepts_hidden_multiplier():
    torch.manual_seed(62)
    cfg = DebugGPT2Config(dropout=0.0)
    model = LadderSideGPT2(
        GPT2LMHeadModel(cfg),
        reduction_factor=8,
        num_side_layers=2,
        side_block_type="drn_hybrid_attn",
        drn_iter=2,
        drn_hidden_multiplier=8,
    )

    assert model.drn_hidden_multiplier == 8
    assert model.collect_ladder_diagnostics()["drn_hidden_multiplier"] == 8
    for mlp in model.drn_mlps():
        assert mlp.block.layer_dims[0] == 8 * model.d_side


def test_post_drn_residual_drn_input_excludes_current_tap():
    torch.manual_seed(7)
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(
        base,
        reduction_factor=8,
        num_side_layers=2,
        side_block_type="drn_pure",
        ladder_injection_mode="post_drn_residual",
        drn_iter=2,
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
    captured_inputs = []

    def _capture(_module, inputs, _output):
        captured_inputs.append(inputs[0].detach().clone())

    handle = model.side_blocks[0].drn.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model.down_projs[1].weight.zero_()
            model.down_projs[1].bias.zero_()
        model(input_ids, targets=input_ids)
        drn_input_without_tap = captured_inputs[-1]

        with torch.no_grad():
            model.down_projs[1].bias.copy_(
                torch.arange(model.d_side, dtype=model.down_projs[1].bias.dtype)
            )
        model(input_ids, targets=input_ids)
        drn_input_with_changed_tap = captured_inputs[-1]
    finally:
        handle.remove()

    torch.testing.assert_close(drn_input_without_tap, drn_input_with_changed_tap)
