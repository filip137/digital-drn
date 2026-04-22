from pathlib import Path

import pytest

from digital_drn.app.trex_cli import main as trex_scp_main
from digital_drn.app.trex_perfect_diode_run_cli import main as trex_perfect_diode_run_main
from digital_drn.app.trex_run_cli import main as trex_run_main


def test_trex_scp_cli_dry_run_prints_ssh_and_scp_commands(capsys, monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "demo.yaml"
    config_path.write_text("defaults: []\n")

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("dry-run should not execute subprocesses")

    monkeypatch.setattr("digital_drn.app.trex_cli.subprocess.run", _unexpected_run)

    exit_code = trex_scp_main(
        [
            "--config-name",
            "demo",
            "--config-dir",
            str(config_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "ssh filip@trex 'mkdir -p /home/filip/digital_drn/hydra_conf'" in captured.out
    assert f"scp {config_path.resolve()} filip@trex:/home/filip/digital_drn/hydra_conf/demo.yaml" in captured.out


def test_trex_scp_cli_executes_mkdir_then_scp(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "demo.yaml"
    config_path.write_text("defaults: []\n")

    calls: list[list[str]] = []

    def _record_run(command, check):
        assert check is True
        calls.append(list(command))
        return None

    monkeypatch.setattr("digital_drn.app.trex_cli.subprocess.run", _record_run)

    exit_code = trex_scp_main(
        [
            "--config-name",
            "demo",
            "--config-dir",
            str(config_dir),
            "--host",
            "gpu-box",
            "--user",
            "alice",
            "--remote-root",
            "/srv/digital_drn",
        ]
    )
    assert exit_code == 0

    assert calls == [
        ["ssh", "alice@gpu-box", "mkdir -p /srv/digital_drn/hydra_conf"],
        ["scp", str(config_path.resolve()), "alice@gpu-box:/srv/digital_drn/hydra_conf/demo.yaml"],
    ]


def test_trex_run_cli_dry_run_prints_forwarded_train_command(capsys, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("defaults: []\n")

    exit_code = trex_run_main(
        [
            "--config-name",
            "demo",
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
        f"+ digital-drn-train --config-name demo --config-dir {config_dir} --epochs 3 --batch-size 8"
        in captured.out
    )


def test_trex_run_cli_calls_train_main_with_forwarded_args(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("defaults: []\n")

    seen: list[list[str]] = []

    def _record_train(argv):
        seen.append(list(argv))
        return 0

    monkeypatch.setattr("digital_drn.app.trex_run_cli.train_main", _record_train)

    exit_code = trex_run_main(
        [
            "--config-name",
            "demo",
            "--config-dir",
            str(config_dir),
            "--",
            "--epochs",
            "5",
        ]
    )
    assert exit_code == 0
    assert seen == [["--config-name", "demo", "--config-dir", str(config_dir), "--epochs", "5"]]


def test_trex_run_cli_rejects_duplicate_reserved_args(tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("defaults: []\n")

    with pytest.raises(ValueError, match="--config-name"):
        trex_run_main(
            [
                "--config-name",
                "demo",
                "--config-dir",
                str(config_dir),
                "--",
                "--config-name",
                "other",
            ]
        )


def test_trex_perfect_diode_run_cli_dry_run_forces_output_dir_and_non_linearity(
    capsys,
    monkeypatch,
    tmp_path: Path,
):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("defaults: []\n")

    seen: dict[str, object] = {}

    def _load_config(**kwargs):
        seen["load_kwargs"] = kwargs
        return {
            "algorithm": {"name": "ep"},
            "trainer": {"epochs": 20, "optimizer": {"lr": 1.0e-4}},
            "data": {"name": "cifar10", "config": {"batch_size": 16}},
            "config": {"output_dir": "simulation_results"},
            "model": {
                "name": "demo_model",
                "config": {"drn_non_linearity": "linear"},
                "blocks_config": [
                    {"drn": {"drn_non_linearity": "linear"}},
                    {"drn": {}},
                ],
            },
        }

    class _DummyModel:
        blocks = [object(), object()]

    class _DummyTrainer:
        device = "cpu"

    def _build_model(config):
        seen["model_config"] = config
        return _DummyModel()

    def _build_trainer(model, config):
        seen["trainer_config"] = config
        return _DummyTrainer()

    monkeypatch.setattr(
        "digital_drn.app.trex_perfect_diode_run_cli.load_experiment_config",
        _load_config,
    )
    monkeypatch.setattr(
        "digital_drn.app.trex_perfect_diode_run_cli.build_model_from_config",
        _build_model,
    )
    monkeypatch.setattr(
        "digital_drn.app.trex_perfect_diode_run_cli.build_trainer_from_config",
        _build_trainer,
    )

    exit_code = trex_perfect_diode_run_main(
        [
            "--config-name",
            "demo",
            "--config-dir",
            str(config_dir),
            "--dry-run",
            "--",
            "--epochs",
            "3",
        ]
    )
    assert exit_code == 0

    forced = seen["model_config"]
    assert forced["config"]["output_dir"] == "simulation_results/trex_simulations"
    assert forced["model"]["config"]["drn_non_linearity"] == "perfect_diode"
    assert forced["model"]["blocks_config"][0]["drn"]["drn_non_linearity"] == "perfect_diode"
    assert forced["model"]["blocks_config"][1]["drn"]["drn_non_linearity"] == "perfect_diode"

    captured = capsys.readouterr()
    assert "output_dir=simulation_results/trex_simulations" in captured.out
    assert "forced drn_non_linearity=perfect_diode" in captured.out


def test_trex_perfect_diode_run_cli_rejects_reserved_checkpoint_dir(tmp_path: Path):
    config_dir = tmp_path / "hydra_conf"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("defaults: []\n")

    with pytest.raises(ValueError, match="--checkpoint-dir"):
        trex_perfect_diode_run_main(
            [
                "--config-name",
                "demo",
                "--config-dir",
                str(config_dir),
                "--",
                "--checkpoint-dir",
                "other",
            ]
        )
