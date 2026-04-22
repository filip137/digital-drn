from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .cli import _print_run_summary
from ..training.config import SchedulerConfig
from ..training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
)
from ..training.trainer import build_scheduler


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Saved experiment config not found: {config_path}")
    with config_path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(data).__name__}.")
    return data


def _resolve_checkpoint_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    else:
        checkpoint_path = (run_dir / args.checkpoint_name).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    run_cfg = config.setdefault("config", {})
    trainer_cfg = config.setdefault("trainer", {})
    data_cfg = config.setdefault("data", {}).setdefault("config", {})

    if args.device is not None:
        run_cfg["device"] = args.device
    if args.checkpoint_dir is not None:
        run_cfg["output_dir"] = args.checkpoint_dir
    if args.save_last:
        trainer_cfg["save_last"] = True

    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.train_subset is not None:
        data_cfg["train_subset"] = args.train_subset
    if args.test_subset is not None:
        data_cfg["test_subset"] = args.test_subset
    return config


def _make_dummy_scheduler_state(
    *,
    trainer,
    checkpoint: dict[str, Any],
    lr_epoch: int,
) -> tuple[list[float], dict[str, Any] | None]:
    history = checkpoint.get("history") or {}
    logged_lrs = list(history.get("learning_rate") or [])
    if lr_epoch < 1:
        raise ValueError("--lr-epoch must be >= 1.")
    if logged_lrs and lr_epoch > len(logged_lrs):
        raise ValueError(
            f"--lr-epoch={lr_epoch} exceeds saved history length {len(logged_lrs)}."
        )

    if trainer.scheduler is None:
        return [float(group["lr"]) for group in trainer.optimizer.param_groups], None

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is None:
        raise ValueError("Checkpoint does not contain scheduler_state_dict.")

    base_lrs = list(scheduler_state.get("base_lrs") or [])
    if not base_lrs:
        raise ValueError("Checkpoint scheduler_state_dict is missing base_lrs.")

    scheduler_cfg = trainer.config.scheduler
    if scheduler_cfg is None or scheduler_cfg.name is None:
        raise ValueError("Trainer does not expose a scheduler config.")

    dummy_params = [torch.nn.Parameter(torch.zeros(())) for _ in base_lrs]
    dummy_groups = [
        {"params": [param], "lr": float(base_lr), "initial_lr": float(base_lr)}
        for param, base_lr in zip(dummy_params, base_lrs)
    ]
    dummy_optimizer = torch.optim.SGD(dummy_groups, lr=float(base_lrs[0]))
    dummy_scheduler = build_scheduler(dummy_optimizer, scheduler_cfg)
    if dummy_scheduler is None:
        raise ValueError("Failed to rebuild scheduler for LR reset.")

    # history.learning_rate[k] is the LR used during epoch (k + 1), which corresponds
    # to the scheduler after k completed steps.
    for _ in range(lr_epoch - 1):
        dummy_optimizer.step()
        dummy_scheduler.step()

    lrs = [float(group["lr"]) for group in dummy_optimizer.param_groups]
    if logged_lrs:
        expected = float(logged_lrs[lr_epoch - 1])
        if abs(lrs[0] - expected) > 1.0e-12:
            raise ValueError(
                f"Reconstructed LR {lrs[0]:.16g} does not match saved history {expected:.16g} at epoch {lr_epoch}."
            )
    return lrs, dummy_scheduler.state_dict()


def _reset_trainer_to_lr_epoch(*, trainer, checkpoint: dict[str, Any], lr_epoch: int) -> list[float]:
    lrs, scheduler_state = _make_dummy_scheduler_state(
        trainer=trainer,
        checkpoint=checkpoint,
        lr_epoch=lr_epoch,
    )
    for group, lr in zip(trainer.optimizer.param_groups, lrs):
        group["lr"] = float(lr)
        if "initial_lr" in group:
            # Keep the scheduler's original base learning rates unchanged.
            group["initial_lr"] = float(group["initial_lr"])

    if trainer.scheduler is not None and scheduler_state is not None:
        trainer.scheduler.load_state_dict(scheduler_state)
        for group, lr in zip(trainer.optimizer.param_groups, lrs):
            group["lr"] = float(lr)
    return lrs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-resume-epoch-lr",
        description="Resume a saved run while resetting optimizer/scheduler to the learning-rate state from a chosen source epoch.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pt")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--lr-epoch", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--extra-epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.epochs is not None and args.extra_epochs is not None:
        raise ValueError("Pass only one of --epochs or --extra-epochs.")
    if args.epochs is None and args.extra_epochs is None:
        raise ValueError("Pass one of --epochs or --extra-epochs.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    checkpoint_path = _resolve_checkpoint_path(args, run_dir)
    config = _apply_overrides(_load_run_config(run_dir), args)
    if args.dry_run:
        config.setdefault("config", {})["save"] = False

    if args.print_config:
        print(json.dumps(config, indent=2))

    _print_run_summary(config, dry_run=args.dry_run)
    print(f"resume: source_run={run_dir}")
    print(f"resume: source_checkpoint={checkpoint_path}")

    model = build_model_from_config(config)
    trainer = build_trainer_from_config(model, config)
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    lrs = _reset_trainer_to_lr_epoch(trainer=trainer, checkpoint=checkpoint, lr_epoch=args.lr_epoch)

    if args.extra_epochs is not None:
        trainer.config.epochs = trainer.history.epochs_completed + args.extra_epochs
    else:
        trainer.config.epochs = int(args.epochs)

    print(
        f"resume: loaded {checkpoint_path} at epoch={trainer.history.epochs_completed} "
        f"step={trainer.history.steps_completed}"
    )
    print(f"resume: reset_lr_epoch={args.lr_epoch} param_group_lrs={[f'{lr:.12g}' for lr in lrs]}")
    print(f"resume: target_total_epochs={trainer.config.epochs}")

    if args.dry_run:
        print(f"dry_run: built {len(model.blocks)} block(s) on device={trainer.device}")
        print(
            "dry_run: would continue under output_dir="
            f"{config.get('config', {}).get('output_dir')}"
        )
        trainer.close()
        return 0

    train_loader, eval_loader = build_dataloaders_from_config(config, download=args.download)
    history = trainer.fit(train_loader, eval_loader, max_steps=args.max_steps)

    if eval_loader is not None:
        eval_metrics = trainer.evaluate(eval_loader)
        print(
            f"final: train_loss={history.train_loss[-1]:.4f} "
            f"train_acc={history.train_accuracy[-1]:.4f} "
            f"val_loss={eval_metrics.loss:.4f} val_acc={eval_metrics.accuracy:.4f}"
        )
    else:
        print(
            f"final: train_loss={history.train_loss[-1]:.4f} "
            f"train_acc={history.train_accuracy[-1]:.4f}"
        )

    checkpoint_dir = trainer.config.checkpoint_dir
    if checkpoint_dir is not None:
        print(f"checkpoints: {Path(checkpoint_dir).resolve()}")
    trainer.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
