import math

import torch
import torch.nn as nn

from digital_drn import (
    DRNGPTConfig,
    SmallDRNGPT,
    transformer_drn_ep_diagnostic,
)


def _tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "seq_len": 6,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 2,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "drn_num_iterations": 4,
        "drn_non_linearity": "linear",
        "drn_ff_activation": "identity",
        "drn_signed_drive": True,
        "drn_weight_gains": 0.1,
        "drn_bias_gain": 0.0,
    }
    values.update(overrides)
    return DRNGPTConfig(**values)


def test_transformer_drn_ep_diagnostic_collects_tokenwise_mlp_groups():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = SmallDRNGPT(cfg).set_device(torch.device("cpu"))
    model.enable_resistive_grad_()

    input_ids = torch.randint(0, cfg.vocab_size, (3, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (3, cfg.seq_len))

    diagnostic = transformer_drn_ep_diagnostic(
        model,
        input_ids,
        targets,
        criterion=nn.CrossEntropyLoss(),
        beta=1.0e-2,
        reset=True,
        num_iterations=4,
    )

    assert diagnostic.logits.shape == (3, cfg.seq_len, cfg.vocab_size)
    assert len(diagnostic.blocks) == cfg.n_layers
    assert set(diagnostic.bp_groups) == set(diagnostic.ep_groups)
    assert "blocks.0.mlp/ff" in diagnostic.bp_groups
    assert "blocks.0.mlp/drive" in diagnostic.bp_groups
    assert "blocks.0.mlp/drn" in diagnostic.bp_groups

    for block_result in diagnostic.blocks:
        assert block_result.output_cotangent.shape == block_result.free_cache.output.shape
        assert block_result.digital.delta_h_prev.shape == block_result.free_cache.h_prev.shape
        assert len(block_result.ep.param_grads) == len(block_result.mlp.block.resistive_params())
        assert all(
            grad.shape == param.state.shape
            for grad, param in zip(block_result.ep.param_grads, block_result.mlp.block.resistive_params())
        )

    assert math.isfinite(diagnostic.loss)
    assert math.isfinite(diagnostic.overall_metrics["bp_norm"])
    assert math.isfinite(diagnostic.overall_metrics["ep_norm"])
    assert diagnostic.overall_metrics["bp_norm"] > 0.0
    assert diagnostic.overall_metrics["ep_norm"] > 0.0
    assert math.isfinite(diagnostic.displacement["mean_relative_disp"])
