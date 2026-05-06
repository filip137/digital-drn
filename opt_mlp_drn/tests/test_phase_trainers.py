import json
import subprocess
import sys


def test_progressive_debug_cli_writes_checkpoint(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opt_mlp_drn.progressive_train",
            "--debug",
            "--tokenizer",
            "char",
            "--layers",
            "all",
            "--max_layer",
            "0",
            "--block_size",
            "16",
            "--batch_size",
            "2",
            "--steps_per_layer",
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
    run_dir = next(tmp_path.glob("opt_mlp_drn_progressive_*"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["layers"] == [0]
    assert (run_dir / "layer_0" / "checkpoint_last.pt").exists()
    assert (run_dir / "progressive_single_blocks" / "layer_0" / "checkpoint_last.pt").exists()
    assert metadata["single_block_checkpoint_paths"]["0"].endswith("checkpoint_last.pt")
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["layers"][0]["hidden_1_rel_rms"] >= 0.0


def test_joint_debug_cli_writes_checkpoint(tmp_path):
    split_text = (
        "First Citizen: Before we proceed any further, hear me speak.\n"
        "All: Speak, speak.\n"
        "This synthetic split is intentionally tiny for a debug smoke test.\n"
    ) * 8
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    test_path = tmp_path / "test.txt"
    train_path.write_text(split_text, encoding="utf-8")
    val_path.write_text(split_text, encoding="utf-8")
    test_path.write_text(split_text, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "opt_mlp_drn.joint_train",
            "--debug",
            "--tokenizer",
            "char",
            "--train_data",
            str(train_path),
            "--val_data",
            str(val_path),
            "--test_data",
            str(test_path),
            "--replace_mlp_layers",
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
            "--early_stopping_metric",
            "logit_kl",
            "--patience",
            "1",
            "--distill_objective",
            "logit_kl",
            "--distill_beta",
            "1.0",
            "--train_lm_head",
            "--drn_iter",
            "1",
            "--output_dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        check=True,
    )
    run_dir = next(tmp_path.glob("opt_mlp_drn_joint_*"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["replace_mlp_layers"] == "all"
    assert metadata["replaced_layer_indices"] == [0, 1, 2]
    assert metadata["distill_objective"] == "logit_kl"
    assert metadata["train_lm_head"] is True
    assert metadata["lm_head_untied_from_embeddings"] is True
    assert metadata["trainable_embedding_params"] == 0
    assert metadata["trainable_lm_head_params"] > 0
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["checkpoint_paths"]["last"].endswith("checkpoint_last.pt")
    assert final["hidden_max_rel_rms"] >= 0.0
    assert final["student_ce_loss"] >= 0.0
    assert final["teacher_ce_loss"] >= 0.0
    assert final["student_ppl"] >= 1.0
    assert final["teacher_ppl"] >= 1.0
    assert final["test_student_ppl"] >= 1.0
    assert final["distill_kl_loss"] >= 0.0
    assert final["shifted_student_ce_loss"] >= 0.0
    assert final["checkpoint_paths"]["best"].endswith("checkpoint_best.pt")


def test_experiment_matrix_lists_and_writes_plan(tmp_path):
    plan_path = tmp_path / "matrix.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "opt_mlp_drn.experiment_matrix", "--list", "--write_plan", str(plan_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dataset_real_1M" in result.stdout
    rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
    assert {row["dataset_source"] for row in rows} >= {"real", "synthetic", "mixed"}
    assert {row["token_budget"] for row in rows} >= {"1M", "10M", "50M"}
    assert any(row["loss"] == "hidden_kl_ce" for row in rows)
