import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from opt_mlp_drn import OPTDRNDecoderLayer, OPTMLPDRNForCausalLM, parse_layer_indices
from opt_mlp_drn.checkpoints import load_single_block_checkpoint_into_layer

transformers = pytest.importorskip("transformers")
OPTConfig = transformers.OPTConfig


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


def _tiny_model(layers="last:1", num_layers=3):
    return OPTMLPDRNForCausalLM.from_config(
        _tiny_config(num_layers=num_layers),
        replace_mlp_layers=layers,
        drn_iter=1,
        signed_drive=True,
    )


def test_last_one_resolves_to_final_opt_layer():
    assert parse_layer_indices("last:1", 12) == [11]
    assert parse_layer_indices("last:2", 12) == [10, 11]
    assert parse_layer_indices("0,2-3", 12) == [0, 2, 3]


def test_only_selected_final_mlp_is_replaced_and_teacher_is_frozen():
    model = _tiny_model(layers="last:1", num_layers=3)

    assert not isinstance(model.base.model.decoder.layers[0], OPTDRNDecoderLayer)
    assert not isinstance(model.base.model.decoder.layers[1], OPTDRNDecoderLayer)
    final_layer = model.base.model.decoder.layers[2]
    assert isinstance(final_layer, OPTDRNDecoderLayer)
    assert all(not param.requires_grad for param in final_layer.fc1.parameters())
    assert all(not param.requires_grad for param in final_layer.fc2.parameters())

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(".drn_mlp." in name or name.endswith(".drn_output_gain") for name in trainable_names)


def test_residual_distillation_loss_backprops_only_to_drn():
    torch.manual_seed(12)
    model = _tiny_model(layers="last:1", num_layers=3)
    model.train()
    input_ids = torch.randint(0, 128, (2, 8))

    result = model.distillation_loss(input_ids)
    loss = result["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert result["metrics"]["distill_loss"] >= 0.0
    assert result["metrics"]["delta_mse"] >= 0.0
    assert result["metrics"]["post_residual_mse"] >= 0.0

    frozen_grads = [
        param.grad
        for name, param in model.named_parameters()
        if ".drn_mlp." not in name and not name.endswith(".drn_output_gain")
    ]
    assert all(grad is None for grad in frozen_grads)

    drn_param_grads = [
        param.grad
        for name, param in model.named_parameters()
        if (".drn_mlp." in name or name.endswith(".drn_output_gain")) and param.requires_grad
    ]
    assert drn_param_grads
    assert any(grad is not None for grad in drn_param_grads)

    resistive_grads = [tensor.grad for _name, tensor in model.named_resistive_parameters()]
    assert resistive_grads
    assert any(grad is not None for grad in resistive_grads)




def test_inactive_replacement_layer_uses_teacher_mlp_path():
    torch.manual_seed(13)
    model = _tiny_model(layers="last:1", num_layers=3)
    model.set_active_replacement_layers([])
    model.set_replacement_probability(1.0)
    input_ids = torch.randint(0, 128, (2, 8))

    model(input_ids)

    cache = model.replaced_layers()[0].distillation_cache()
    assert cache.active_replacement is False
    assert cache.used_student_mlp is False
    assert cache.replacement_probability == 1.0

def test_single_block_checkpoint_loader_restores_resistive_tensors(tmp_path):
    model = _tiny_model(layers="last:1", num_layers=3)
    layer = model.replaced_layers()[0]
    state = {f"drn_mlp.{name}": tensor.detach().clone() for name, tensor in layer.drn_mlp.state_dict().items()}
    state["output_gain"] = torch.tensor(1.75)
    expected = {
        name: torch.full_like(tensor.detach(), 0.02 + 0.001 * idx)
        for idx, (name, tensor) in enumerate(layer.drn_mlp.named_resistive_parameters())
    }
    checkpoint_path = tmp_path / "single_block.pt"
    torch.save({"model": state, "resistive_parameters": expected}, checkpoint_path)

    with torch.no_grad():
        for _name, tensor in layer.drn_mlp.named_resistive_parameters():
            tensor.zero_()

    load_single_block_checkpoint_into_layer(checkpoint_path, layer)

    actual = dict(layer.drn_mlp.named_resistive_parameters())
    assert set(actual) == set(expected)
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor)
    torch.testing.assert_close(layer.drn_output_gain.detach(), torch.tensor(1.75))


def test_single_block_checkpoint_loader_accepts_legacy_amp_keys(tmp_path):
    model = _tiny_model(layers="last:1", num_layers=3)
    layer = model.replaced_layers()[0]
    state = {f"drn_mlp.{name}": tensor.detach().clone() for name, tensor in layer.drn_mlp.state_dict().items()}
    expected_voltage = torch.tensor(0.25)
    expected_current = torch.tensor(-0.5)
    state["drn_mlp.block.voltage_amp_raw"] = expected_voltage
    state["drn_mlp.block.current_amp_raw"] = expected_current
    state.pop("drn_mlp.block._voltage_amp_raw")
    state.pop("drn_mlp.block._current_amp_raw")
    checkpoint_path = tmp_path / "legacy_single_block.pt"
    torch.save({"model": state}, checkpoint_path)

    load_single_block_checkpoint_into_layer(checkpoint_path, layer)

    loaded = layer.drn_mlp.state_dict()
    torch.testing.assert_close(loaded["block._voltage_amp_raw"], expected_voltage)
    torch.testing.assert_close(loaded["block._current_amp_raw"], expected_current)


def test_debug_cli_residual_distill_writes_metadata_and_checkpoint(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opt_mlp_drn.train",
            "--debug",
            "--tokenizer",
            "char",
            "--block_size",
            "16",
            "--batch_size",
            "2",
            "--objective",
            "residual_distill",
            "--distill_steps",
            "1",
            "--eval_interval",
            "1",
            "--eval_iters",
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
    run_dirs = list(tmp_path.glob("opt_mlp_drn_*"))
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["objective"] == "residual_distill"
    assert metadata["replace_mlp_layers"] == "last:1"
    assert metadata["replaced_layer_indices"] == [1]
    assert metadata["trainable_params"] > 0
    assert (run_dirs[0] / "checkpoint_distilled.pt").exists()


def test_cached_facebook_opt_125m_forward_if_available():
    cache_root = Path.home() / ".cache/huggingface/hub/models--facebook--opt-125m/snapshots"
    snapshots = sorted(cache_root.glob("*")) if cache_root.exists() else []
    if not snapshots:
        pytest.skip("facebook/opt-125m is not cached locally.")

    model = OPTMLPDRNForCausalLM.from_pretrained(
        str(snapshots[-1]),
        replace_mlp_layers="last:1",
        drn_iter=1,
    )
    input_ids = torch.tensor([[2, 10, 11, 12]], dtype=torch.long)
    out = model(input_ids, targets=input_ids)
    assert out["logits"].shape == (1, 4, 50272)
    assert torch.isfinite(out["loss"])
