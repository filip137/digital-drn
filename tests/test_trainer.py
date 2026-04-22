from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import digital_drn.training.trainer as trainer_module
from digital_drn import BPTrainer, OptimizerConfig, SequentialDigitalDRNNet, TrainerConfig


def _make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _make_binary_data(num_samples: int, input_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = 0.1 * torch.randn(num_samples, input_dim)
    y = (x.sum(dim=1) > 0.0).long()
    return x, y


def _make_model(input_dim: int = 3, seed: int | None = None) -> SequentialDigitalDRNNet:
    if seed is not None:
        torch.manual_seed(seed)
    return SequentialDigitalDRNNet(
        input_dim=input_dim,
        block_configs=[{"layer_dims": [4, 2], "weight_gains": [0.1], "bias_gain": 0.0}],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        weight_min=0.0,
        weight_max=1.0,
    )


def test_trainer_fit_and_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(3)
    train_x = torch.randn(8, 3)
    train_y = torch.randint(0, 2, (8,))
    val_x = torch.randn(4, 3)
    val_y = torch.randint(0, 2, (4,))
    train_loader = _make_loader(train_x, train_y, batch_size=8, shuffle=False)
    val_loader = _make_loader(val_x, val_y, batch_size=4, shuffle=False)

    config = TrainerConfig(
        epochs=1,
        optimizer=OptimizerConfig(name="sgd", lr=0.1),
        checkpoint_dir=tmp_path,
        device="cpu",
        seed=3,
        log_every=0,
        eval_every=1,
        save_every=1,
        save_best=True,
        save_last=True,
        train_num_iterations=2,
        eval_num_iterations=2,
    )
    trainer = BPTrainer(_make_model(seed=3), config)
    history = trainer.fit(train_loader, val_loader, max_steps=1)

    assert len(history.train_loss) == 1
    assert len(history.train_accuracy) == 1
    assert len(history.val_loss) == 1
    assert len(history.val_accuracy) == 1
    assert history.epochs_completed == 1
    assert history.steps_completed == 1
    assert history.best_val_accuracy is not None

    assert (tmp_path / "trainer_config.json").exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "checkpoint_last.pt").exists()
    assert (tmp_path / "checkpoint_best.pt").exists()
    assert (tmp_path / "checkpoint_epoch_0001.pt").exists()
    assert any(path.name.startswith("events.out.tfevents.") for path in tmp_path.iterdir())

    expected_weight = trainer.model.blocks[0].energy.dense_weights[0].state.detach().clone()
    expected_drive_scale = trainer.model.blocks[0].drive_scale.detach().clone()

    restored_trainer = BPTrainer(_make_model(seed=11), TrainerConfig(device="cpu", seed=123, log_every=0))
    checkpoint = restored_trainer.load_checkpoint(tmp_path / "checkpoint_last.pt")

    assert "resistive_state" in checkpoint
    assert restored_trainer.history.steps_completed == history.steps_completed
    assert restored_trainer.history.epochs_completed == history.epochs_completed
    assert torch.allclose(
        restored_trainer.model.blocks[0].energy.dense_weights[0].state.detach(),
        expected_weight,
    )
    assert torch.allclose(
        restored_trainer.model.blocks[0].drive_scale.detach(),
        expected_drive_scale,
    )
    trainer.close()
    restored_trainer.close()


def test_trainer_max_steps_stops_early():
    torch.manual_seed(3)
    train_x, train_y = _make_binary_data(20, 3)
    train_loader = _make_loader(train_x, train_y, batch_size=5)

    trainer = BPTrainer(
        _make_model(seed=3),
        TrainerConfig(
            epochs=5,
            optimizer=OptimizerConfig(name="adam", lr=1.0e-3),
            device="cpu",
            seed=3,
            log_every=0,
            eval_every=0,
            save_every=0,
            save_best=False,
            train_num_iterations=2,
            eval_num_iterations=2,
        ),
    )
    history = trainer.fit(train_loader, max_steps=3)

    assert history.steps_completed == 3
    assert history.epochs_completed == 1


def test_restore_rng_state_accepts_non_byte_tensor_state():
    cpu_state = torch.get_rng_state().to(dtype=torch.int64)
    BPTrainer._restore_rng_state({"torch": cpu_state})
    restored = torch.get_rng_state()
    assert restored.dtype == torch.uint8


def test_trainer_logs_block_diagnostics_to_writer(monkeypatch, tmp_path: Path):
    class FakeWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), int(step)))

        def flush(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(trainer_module, "SummaryWriter", FakeWriter)

    torch.manual_seed(7)
    train_x, train_y = _make_binary_data(8, 3)
    val_x, val_y = _make_binary_data(4, 3)
    train_loader = _make_loader(train_x, train_y, batch_size=4, shuffle=False)
    val_loader = _make_loader(val_x, val_y, batch_size=4, shuffle=False)

    trainer = BPTrainer(
        _make_model(seed=7),
        TrainerConfig(
            epochs=1,
            optimizer=OptimizerConfig(name="adam", lr=1.0e-3),
            checkpoint_dir=tmp_path,
            device="cpu",
            seed=7,
            log_every=0,
            eval_every=1,
            save_every=0,
            save_best=False,
            save_last=False,
            save_events=True,
            train_num_iterations=2,
            eval_num_iterations=2,
        ),
    )

    trainer.fit(train_loader, val_loader)
    tags = {tag for tag, _value, _step in trainer.writer.scalars}

    assert "diagnostics/val/block_0/drive_mean" in tags
    assert "diagnostics/val/block_0/drive_std" in tags
    assert "diagnostics/val/block_0/z1_mean" in tags
    assert "diagnostics/val/block_0/z1_std" in tags
    assert "diagnostics/val/block_0/z_out_mean" in tags
    assert "diagnostics/val/block_0/drive_scale" in tags
    assert "diagnostics/val/block_0/drive_to_z1_rms_ratio" in tags
    assert "weights/block_0/ff/1_weight/mean" in tags
    assert "gradients/block_0/ff/1_weight/mean" in tags
    assert any(tag.startswith("weights/block_0/drn/DenseWeight_") and tag.endswith("/mean") for tag in tags)
    assert any(tag.startswith("gradients/block_0/drn/DenseWeight_") and tag.endswith("/mean") for tag in tags)
    trainer.close()
