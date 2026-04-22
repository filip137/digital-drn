from pathlib import Path

import pytest

from digital_drn.app.cifar10_wider_train_cli import main


def test_cifar10_wider_train_cli_dry_run_prints_forwarded_train_command(capsys, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "cifar10_drn_only_signed_norm_readout_wider.yaml").write_text("defaults: []\n")

    exit_code = main(
        [
            "--config-dir",
            str(config_dir),
            "--dry-run",
            "--",
            "--epochs",
            "3",
            "--batch-size",
            "8",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert (
        f"+ digital-drn-train --config-name cifar10_drn_only_signed_norm_readout_wider "
        f"--config-dir {config_dir} --epochs 3 --batch-size 8"
        in captured.out
    )


def test_cifar10_wider_train_cli_calls_train_main_with_forwarded_args(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "cifar10_drn_only_signed_norm_readout_wider.yaml").write_text("defaults: []\n")

    seen: list[list[str]] = []

    def _record_train(argv):
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("digital_drn.app.cifar10_wider_train_cli.train_main", _record_train)

    exit_code = main(
        [
            "--config-dir",
            str(config_dir),
            "--",
            "--epochs",
            "5",
        ]
    )
    assert exit_code == 0
    assert seen == [[
        "--config-name",
        "cifar10_drn_only_signed_norm_readout_wider",
        "--config-dir",
        str(config_dir),
        "--epochs",
        "5",
    ]]


def test_cifar10_wider_train_cli_rejects_duplicate_reserved_args(tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "cifar10_drn_only_signed_norm_readout_wider.yaml").write_text("defaults: []\n")

    with pytest.raises(ValueError, match="--config-name"):
        main(
            [
                "--config-dir",
                str(config_dir),
                "--",
                "--config-name",
                "other",
            ]
        )
