import json
from pathlib import Path

from digital_drn.app.cli import main
from digital_drn.training.experiment import load_experiment_config


def test_cli_dry_run_with_overrides(capsys):
    exit_code = main(
        [
            "--dry-run",
            "--model",
            "conv_mnist_1block",
            "--epochs",
            "3",
            "--extra-epochs",
            "2",
            "--batch-size",
            "32",
            "--checkpoint-dir",
            "/tmp/digital_drn_test_cli",
            "--resume-from",
            "/tmp/checkpoint_best.pt",
            "--save-last",
            "--print-config",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    config_prefix = "{\n"
    assert config_prefix in captured.out
    assert '"epochs": 3' in captured.out
    assert '"batch_size": 32' in captured.out
    assert '"save_last": true' in captured.out
    assert "dry_run: built 1 block(s)" in captured.out
    assert "dry_run: would resume from /tmp/checkpoint_best.pt" in captured.out


def test_cli_override_path_matches_loaded_config():
    base = load_experiment_config(model="dense_1block")
    assert base["trainer"]["epochs"] == 200

    exit_code = main(["--dry-run", "--model", "dense_1block", "--epochs", "5"])
    assert exit_code == 0


def test_cli_extra_epochs_requires_resume():
    try:
        main(["--extra-epochs", "10"])
    except ValueError as exc:
        assert "--extra-epochs requires --resume-from." in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError when using --extra-epochs without --resume-from.")


def test_cli_resume_uses_saved_experiment_config(capsys, tmp_path: Path):
    config = load_experiment_config(model="dense_1block")
    config["trainer"]["epochs"] = 7
    config["data"]["config"]["batch_size"] = 11

    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    with (run_dir / "experiment_config.json").open("w") as handle:
        json.dump(config, handle)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    checkpoint_path.write_bytes(b"stub")

    exit_code = main(["--dry-run", "--resume-from", str(checkpoint_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "epochs=7" in captured.out
    assert "batch_size=11" in captured.out
    assert "resume: using config snapshot" in captured.out
