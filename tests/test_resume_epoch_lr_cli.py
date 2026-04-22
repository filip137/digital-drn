import json
from pathlib import Path

import torch

from digital_drn.app.resume_epoch_lr_cli import _make_dummy_scheduler_state, main
from digital_drn.training.config import OptimizerConfig, SchedulerConfig, TrainerConfig
from digital_drn.training.trainer import TrainHistory


class _DummyModel:
    blocks = [object(), object()]


class _DummyTrainer:
    def __init__(self) -> None:
        self.device = "cpu"
        self.config = TrainerConfig(
            epochs=200,
            optimizer=OptimizerConfig(name="adam", lr=1.0e-3),
            scheduler=SchedulerConfig(name="cosine", config={"T_max": 200, "eta_min": 1.0e-5}),
            checkpoint_dir="/tmp/resume_epoch_lr_test",
        )
        params = [torch.nn.Parameter(torch.zeros(())) for _ in range(2)]
        self.optimizer = torch.optim.Adam(
            [
                {"params": [params[0]], "lr": 1.0e-3, "initial_lr": 1.0e-3},
                {"params": [params[1]], "lr": 1.0e-4, "initial_lr": 1.0e-4},
            ],
            lr=1.0e-3,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200,
            eta_min=1.0e-5,
        )
        self.history = TrainHistory()

    def load_checkpoint(self, path: Path):
        assert path.exists()
        learning_rate = []
        params = [torch.nn.Parameter(torch.zeros(())) for _ in range(2)]
        opt = torch.optim.Adam(
            [
                {"params": [params[0]], "lr": 1.0e-3, "initial_lr": 1.0e-3},
                {"params": [params[1]], "lr": 1.0e-4, "initial_lr": 1.0e-4},
            ],
            lr=1.0e-3,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1.0e-5)
        for _epoch in range(200):
            learning_rate.append(float(opt.param_groups[0]["lr"]))
            opt.step()
            sched.step()

        self.history = TrainHistory(epochs_completed=196, steps_completed=12345, learning_rate=learning_rate)
        return {
            "history": self.history.to_dict(),
            "scheduler_state_dict": {
                "T_max": 200,
                "eta_min": 1.0e-5,
                "base_lrs": [1.0e-3, 1.0e-4],
                "last_epoch": 195,
                "verbose": False,
                "_step_count": 196,
                "_get_lr_called_within_step": False,
                "_last_lr": [1.152591980210166e-05, 1.0138719982009242e-05],
            },
        }

    def close(self) -> None:
        pass


def test_make_dummy_scheduler_state_matches_epoch_100_history(tmp_path: Path):
    trainer = _DummyTrainer()
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    checkpoint_path.write_bytes(b"stub")
    checkpoint = trainer.load_checkpoint(checkpoint_path)

    lrs, scheduler_state = _make_dummy_scheduler_state(
        trainer=trainer,
        checkpoint=checkpoint,
        lr_epoch=100,
    )
    assert scheduler_state is not None
    assert abs(lrs[0] - trainer.history.learning_rate[99]) < 1.0e-12
    assert abs(lrs[0] - 5.127751220693511e-04) < 1.0e-12


def test_resume_epoch_lr_cli_dry_run_prints_adjusted_lr_epoch(capsys, monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    config = {
        "algorithm": {"name": "bp", "config": {}},
        "trainer": {
            "epochs": 200,
            "criterion": "cross_entropy",
            "optimizer": {"name": "adam", "lr": 1.0e-4, "config": {}},
            "scheduler": {"name": "cosine", "config": {"T_max": 200, "eta_min": 1.0e-5}},
        },
        "data": {"name": "cifar10", "config": {"batch_size": 16}},
        "model": {"name": "demo_model"},
        "config": {"output_dir": "simulation_results"},
    }
    with (run_dir / "experiment_config.json").open("w") as handle:
        json.dump(config, handle)
    (run_dir / "checkpoint_best.pt").write_bytes(b"stub")

    monkeypatch.setattr(
        "digital_drn.app.resume_epoch_lr_cli.build_model_from_config",
        lambda cfg: _DummyModel(),
    )
    monkeypatch.setattr(
        "digital_drn.app.resume_epoch_lr_cli.build_trainer_from_config",
        lambda model, cfg: _DummyTrainer(),
    )

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--lr-epoch",
            "100",
            "--extra-epochs",
            "50",
            "--dry-run",
        ]
    )
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "resume: reset_lr_epoch=100" in output
    assert "0.000512775122" in output
    assert "dry_run: would continue under output_dir=simulation_results" in output
