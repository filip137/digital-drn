import json
import subprocess
import sys

import pytest
import torch

from opt_mlp_drn.calibration import calibrate_teacher
from opt_mlp_drn.cache_activations import CachedLayerActivationDataset, write_activation_cache
from opt_mlp_drn.single_block import build_single_block_drn, optimizer_param_groups, single_block_loss
from opt_mlp_drn.teacher import collect_teacher_layer_activations, default_probe_layers

transformers = pytest.importorskip("transformers")
OPTConfig = transformers.OPTConfig
OPTForCausalLM = transformers.OPTForCausalLM


def _tiny_config(num_layers=3):
    return OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="relu",
        do_layer_norm_before=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        word_embed_proj_dim=32,
    )


def _teacher(num_layers=3):
    model = OPTForCausalLM(_tiny_config(num_layers=num_layers))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def test_default_probe_layers_for_opt_125m_shape():
    assert default_probe_layers(12) == [0, 6, 11]


def test_collect_teacher_layer_activations_shapes_and_residual_identity():
    torch.manual_seed(7)
    teacher = _teacher(num_layers=3)
    input_ids = torch.randint(0, 128, (2, 8))

    activations = collect_teacher_layer_activations(teacher, input_ids, [0, 2])

    assert sorted(activations) == [0, 2]
    for layer_acts in activations.values():
        assert layer_acts.z.shape == (2, 8, 32)
        assert layer_acts.r.shape == (2, 8, 32)
        assert layer_acts.a.shape == (2, 8, 32)
        assert layer_acts.h_next.shape == (2, 8, 32)
        assert layer_acts.next_ln is not None
        assert layer_acts.next_ln.shape == (2, 8, 32)
        torch.testing.assert_close(layer_acts.a + layer_acts.r, layer_acts.h_next, atol=1.0e-5, rtol=1.0e-5)


def test_collect_teacher_layer_activations_raw_mode_skips_mlp_layer_norm():
    torch.manual_seed(7)
    teacher = _teacher(num_layers=3)
    input_ids = torch.randint(0, 128, (2, 8))
    layer_index = 0

    normalized = collect_teacher_layer_activations(teacher, input_ids, [layer_index])[layer_index]
    raw = collect_teacher_layer_activations(
        teacher,
        input_ids,
        [layer_index],
        mlp_input_mode="raw",
    )[layer_index]
    layer = teacher.model.decoder.layers[layer_index]
    expected_raw_r = layer.fc2(layer.activation_fn(layer.fc1(raw.a)))

    torch.testing.assert_close(raw.z, raw.a)
    torch.testing.assert_close(raw.r, expected_raw_r, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(raw.a + raw.r, raw.h_next, atol=1.0e-6, rtol=1.0e-6)
    assert raw.next_ln is not None
    assert not torch.allclose(raw.z, normalized.z)


def test_calibration_records_z_and_r_stats():
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(8)
    teacher = _teacher(num_layers=3)
    x = torch.randint(0, 128, (4, 8))
    y = torch.randint(0, 128, (4, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    payload = calibrate_teacher(
        teacher,
        loader,
        [0, 2],
        device=torch.device("cpu"),
        max_batches=2,
        max_quantile_samples=2048,
    )

    assert payload["num_batches"] == 2
    assert sorted(payload["layers"]) == ["0", "2"]
    for layer_stats in payload["layers"].values():
        for key in ("z", "r"):
            assert layer_stats[key]["num_values"] == 2 * 2 * 8 * 32
            assert layer_stats[key]["q0_999_abs"] >= 0.0


def test_activation_cache_stores_sharded_layer_targets(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(8)
    teacher = _teacher(num_layers=3)
    x = torch.randint(0, 128, (4, 8))
    y = torch.randint(0, 128, (4, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metadata = write_activation_cache(
        teacher=teacher,
        train_loader=loader,
        val_loader=loader,
        layer_indices=[0, 2],
        output_dir=tmp_path,
        device=torch.device("cpu"),
        model_name="debug",
        block_size=8,
        source="unit",
        mlp_input_mode="raw",
        dtype="float32",
    )
    assert metadata["mlp_input_mode"] == "raw"
    assert metadata["splits"]["train"]["num_shards"] == 2

    dataset = CachedLayerActivationDataset(tmp_path, split="train", layer_index=0)
    item = dataset[0]
    assert set(item) == {"z", "r", "a", "h_next", "next_ln"}
    assert item["z"].shape == (8, 32)
    assert item["next_ln"].shape == (8, 32)
    torch.testing.assert_close(item["z"], item["a"])


def test_single_block_local_and_post_losses_backpropagate():
    torch.manual_seed(9)
    teacher = _teacher(num_layers=3)
    input_ids = torch.randint(0, 128, (2, 8))
    layer_index = 2
    activations = collect_teacher_layer_activations(teacher, input_ids, [layer_index])[layer_index]
    block = build_single_block_drn(
        teacher.model.decoder.layers[layer_index],
        input_scale=1.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="random",
    )

    local = single_block_loss(block, teacher, layer_index, activations, objective="local_mlp")
    post = single_block_loss(block, teacher, layer_index, activations, objective="post_residual")
    assert torch.isfinite(local.loss)
    assert torch.isfinite(post.loss)
    torch.testing.assert_close(local.loss.detach(), post.loss.detach(), atol=1.0e-5, rtol=1.0e-5)

    local.loss.backward()
    assert any(param.grad is not None for param in block.parameters() if param.requires_grad)
    assert any(tensor.grad is not None for tensor in block.resistive_param_states())


def test_rigorous_pretrain_loss_adds_next_ln_and_final_logit_terms():
    torch.manual_seed(14)
    teacher = _teacher(num_layers=3)
    input_ids = torch.randint(0, 128, (2, 8))
    layer_index = 2
    activations = collect_teacher_layer_activations(teacher, input_ids, [layer_index])[layer_index]
    block = build_single_block_drn(
        teacher.model.decoder.layers[layer_index],
        input_scale=1.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        drive_architecture="signed_input_free",
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="random",
    )

    result = single_block_loss(
        block,
        teacher,
        layer_index,
        activations,
        objective="rigorous_pretrain",
        alpha_next_ln=0.1,
        alpha_cosine=0.1,
        alpha_norm=0.1,
        alpha_logit_kl=0.01,
        logit_temperature=2.0,
    )

    assert torch.isfinite(result.loss)
    assert result.metrics["next_ln_mse"] >= 0.0
    assert result.metrics["final_logit_kl"] >= 0.0
    result.loss.backward()
    assert any(tensor.grad is not None for tensor in block.resistive_param_states())


def test_local_mlp_cosine_uses_separate_norm_penalty():
    torch.manual_seed(15)
    teacher = _teacher(num_layers=3)
    input_ids = torch.randint(0, 128, (2, 8))
    layer_index = 2
    activations = collect_teacher_layer_activations(teacher, input_ids, [layer_index])[layer_index]
    block = build_single_block_drn(
        teacher.model.decoder.layers[layer_index],
        input_scale=1.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        drive_architecture="signed_input_free",
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="random",
    )

    no_norm = single_block_loss(
        block,
        teacher,
        layer_index,
        activations,
        objective="local_mlp_cosine",
        alpha_cosine=0.1,
        alpha_norm=0.0,
    )
    with_norm = single_block_loss(
        block,
        teacher,
        layer_index,
        activations,
        objective="local_mlp_cosine",
        alpha_cosine=0.1,
        alpha_norm=10.0,
    )

    assert torch.isfinite(with_norm.loss)
    assert with_norm.loss > no_norm.loss
    assert with_norm.metrics["norm_ratio_loss"] >= 0.0


def test_teacher_frontend_init_copies_fc1_when_signed_drive_disabled():
    torch.manual_seed(10)
    teacher = _teacher(num_layers=3)
    layer = teacher.model.decoder.layers[1]
    block = build_single_block_drn(
        layer,
        input_scale=2.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=False,
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="teacher_frontend_scale",
    )

    linear = next(module for module in block.drn.block.ff.modules() if isinstance(module, torch.nn.Linear))
    torch.testing.assert_close(linear.weight, layer.fc1.weight * 2.0)
    torch.testing.assert_close(linear.bias, layer.fc1.bias)


def test_signed_drive_uses_full_teacher_hidden_width_before_mirroring():
    torch.manual_seed(11)
    teacher = _teacher(num_layers=3)
    layer = teacher.model.decoder.layers[1]
    block = build_single_block_drn(
        layer,
        input_scale=2.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="teacher_frontend_scale",
    )

    linear = next(module for module in block.drn.block.ff.modules() if isinstance(module, torch.nn.Linear))
    assert block.hidden_dim == 2 * layer.fc1.out_features
    assert linear.out_features == layer.fc1.out_features
    torch.testing.assert_close(linear.weight, layer.fc1.weight * 2.0)
    torch.testing.assert_close(linear.bias, layer.fc1.bias)


def test_signed_input_free_architecture_injects_signed_input_without_frontend_weights():
    torch.manual_seed(12)
    teacher = _teacher(num_layers=3)
    layer = teacher.model.decoder.layers[1]
    block = build_single_block_drn(
        layer,
        input_scale=1.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        drive_architecture="signed_input_free",
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        init_mode="random",
    )

    assert block.layer_dims == (2 * layer.embed_dim, 2 * layer.fc1.out_features, layer.embed_dim)
    assert block.hidden_dim == 2 * layer.fc1.out_features
    assert not any(isinstance(module, torch.nn.Linear) for module in block.drn.block.ff.modules())
    assert list(block.drn.block.ff.parameters()) == []
    assert block.drn.block._drive_scale_raw.requires_grad

    x = torch.randn(3, layer.embed_dim)
    drive = block.drn.block.ff(x)
    assert drive.shape == (3, 2 * layer.embed_dim)
    torch.testing.assert_close(drive[:, : layer.embed_dim], x)
    torch.testing.assert_close(drive[:, layer.embed_dim :], -x)


def test_optimizer_param_groups_can_use_separate_amp_lr():
    torch.manual_seed(13)
    teacher = _teacher(num_layers=3)
    layer = teacher.model.decoder.layers[1]
    block = build_single_block_drn(
        layer,
        input_scale=1.0,
        output_scale=1.0,
        drn_iter=1,
        signed_drive=True,
        drive_architecture="signed_input_free",
        hidden_multiplier=None,
        weight_gains=0.1,
        bias_gain=0.0,
        init_drive_scale=1.0,
        learn_amplification=True,
        init_mode="random",
    )

    groups = optimizer_param_groups(block, amp_lr=0.01, output_gain_lr=0.2)
    amp_param_ids = {id(param) for param in block.drn.block.amplification_parameters()}
    output_gain_groups = [
        group
        for group in groups
        if any(id(param) == id(block.output_gain) for param in group["params"])
    ]
    amp_groups = [
        group
        for group in groups
        if any(id(param) in amp_param_ids for param in group["params"])
    ]

    assert len(output_gain_groups) == 1
    assert output_gain_groups[0]["lr"] == 0.2
    assert output_gain_groups[0]["weight_decay"] == 0.0
    assert len(amp_groups) == 1
    assert amp_groups[0]["lr"] == 0.01


def test_single_block_cli_writes_metadata_and_checkpoint(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opt_mlp_drn.single_block_train",
            "--debug",
            "--tokenizer",
            "char",
            "--layers",
            "0",
            "--block_size",
            "16",
            "--batch_size",
            "2",
            "--steps",
            "1",
            "--eval_interval",
            "1",
            "--eval_iters",
            "1",
            "--calibration_batches",
            "1",
            "--eval_logit_kl_batches",
            "1",
            "--drn_iter",
            "1",
            "--output_dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("opt_mlp_drn_single_block_*"))
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["layers"] == [0]
    assert metadata["objective"] == "local_mlp"
    assert metadata["checkpoint_paths"]["0"].endswith("checkpoint_last.pt")
    assert metadata["best_checkpoint_paths"]["0"].endswith("checkpoint_best.pt")
    assert (run_dirs[0] / "layer_0" / "checkpoint_last.pt").exists()
    assert (run_dirs[0] / "layer_0" / "checkpoint_best.pt").exists()
    final_metrics = json.loads((run_dirs[0] / "layer_0" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final_metrics["replacement_logit_kl"] >= 0.0
    assert final_metrics["best_checkpoint_path"].endswith("checkpoint_best.pt")


def test_single_block_cli_all_layers_writes_layer_summary(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opt_mlp_drn.single_block_train",
            "--debug",
            "--tokenizer",
            "char",
            "--layers",
            "all",
            "--block_size",
            "16",
            "--batch_size",
            "2",
            "--steps",
            "1",
            "--eval_interval",
            "1",
            "--eval_iters",
            "1",
            "--calibration_batches",
            "1",
            "--drn_iter",
            "1",
            "--output_dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        check=True,
    )
    run_dir = next(tmp_path.glob("opt_mlp_drn_single_block_*"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["layers"] == [0, 1, 2]
    assert (run_dir / "layer_summary.csv").exists()
    summary = json.loads((run_dir / "layer_summary.json").read_text(encoding="utf-8"))
    assert [row["layer"] for row in summary] == [0, 1, 2]
    assert {"q99_abs_error", "saturation_fraction", "last_train_grad_norm"} <= set(summary[0])
