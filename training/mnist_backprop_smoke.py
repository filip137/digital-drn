from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from .config import OptimizerConfig, TrainerConfig
    from ..models.network import SequentialDigitalDRNNet
    from .trainer import BPTrainer
except ImportError:  # pragma: no cover - support direct script invocation
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from digital_drn.training.config import OptimizerConfig, TrainerConfig
    from digital_drn.models.network import SequentialDigitalDRNNet
    from digital_drn.training.trainer import BPTrainer


@dataclass
class MnistSmokeResult:
    train_losses: list[float]
    eval_loss: float
    eval_accuracy: float
    steps: int
    epochs: int
    device: str
    data_root: str
    seed: int
    drive_scales: list[float]
    ff_activation: str | None
    used_conv_ff: bool


def _make_loader(dataset, batch_size: int, subset_size: int | None, shuffle: bool, generator=None):
    if subset_size is not None:
        dataset = Subset(dataset, list(range(min(subset_size, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def run_mnist_backprop_smoke(
    *,
    data_root: str | Path = "~/data",
    device: str | torch.device | None = None,
    train_subset: int = 2048,
    test_subset: int = 512,
    batch_size: int = 128,
    steps: int | None = 20,
    learning_rate: float = 1e-4,
    num_iterations: int = 4,
    seed: int = 0,
    epochs: int = 1,
    ff_activation: str | None = "tanh",
    use_conv_ff: bool = False,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_root = Path(os.path.expandvars(str(data_root))).expanduser()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=False, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        subset_size=train_subset,
        shuffle=True,
        generator=generator,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        subset_size=test_subset,
        shuffle=False,
        generator=generator,
    )

    block_config = {
        "layer_dims": [128, 10],
        "weight_gains": [0.05],
        "bias_gain": 0.0,
    }
    if use_conv_ff:
        block_config.update(
            {
                "ff_input_shape": (1, 28, 28),
                "ff_conv_channels": [32, 64],
                "ff_conv_kernels": [3, 3],
                "ff_conv_paddings": [1, 1],
                "ff_conv_pool_kernels": [2, 2],
            }
        )

    model = SequentialDigitalDRNNet(
        input_dim=28 * 28,
        block_configs=[block_config],
        num_iterations=num_iterations,
        mode="asynchronous",
        ff_activation=ff_activation,
        non_linearity="linear",
        weight_min=0.0,
        weight_max=1.0,
    )
    trainer = BPTrainer(
        model,
        TrainerConfig(
            epochs=epochs,
            optimizer=OptimizerConfig(name="adam", lr=learning_rate),
            device=device,
            seed=seed,
            train_num_iterations=num_iterations,
            eval_num_iterations=num_iterations,
            train_reset_state=True,
            eval_reset_state=True,
            log_every=0,
            eval_every=1,
            save_every=0,
            save_best=False,
        ),
    )
    history = trainer.fit(train_loader, test_loader, max_steps=steps)
    eval_metrics = trainer.evaluate(test_loader)

    return MnistSmokeResult(
        train_losses=history.train_loss,
        eval_loss=eval_metrics.loss,
        eval_accuracy=eval_metrics.accuracy,
        steps=history.steps_completed,
        epochs=epochs,
        device=str(trainer.device),
        data_root=str(data_root),
        seed=seed,
        drive_scales=[float(block.drive_scale.detach().cpu()) for block in trainer.model.blocks],
        ff_activation=ff_activation,
        used_conv_ff=use_conv_ff,
    )


if __name__ == "__main__":
    result = run_mnist_backprop_smoke()
    print(
        {
            "train_losses": [round(loss, 4) for loss in result.train_losses],
            "eval_loss": round(result.eval_loss, 4),
            "eval_accuracy": round(result.eval_accuracy, 4),
            "steps": result.steps,
            "epochs": result.epochs,
            "device": result.device,
            "data_root": result.data_root,
            "seed": result.seed,
            "drive_scales": [round(scale, 4) for scale in result.drive_scales],
            "ff_activation": result.ff_activation,
            "used_conv_ff": result.used_conv_ff,
        }
    )
