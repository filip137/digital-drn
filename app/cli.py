from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    load_experiment_config,
)


def _load_resume_config(resume_from: str | Path) -> dict[str, Any] | None:
    resume_path = Path(resume_from).expanduser().resolve()
    config_path = resume_path.parent / "experiment_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(data).__name__}.")
    return data


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-train",
        description="Train digital_drn models from the YAML config tree.",
    )
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--trainer", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--extra-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--train-num-iterations", type=int, default=None)
    parser.add_argument("--eval-num-iterations", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))

    run_cfg = config.setdefault("config", {})
    trainer_cfg = config.setdefault("trainer", {})
    optimizer_cfg = trainer_cfg.setdefault("optimizer", {})
    data_cfg = config.setdefault("data", {}).setdefault("config", {})

    if args.device is not None:
        run_cfg["device"] = args.device
    if args.seed is not None:
        run_cfg["seed"] = args.seed
    if args.deterministic:
        run_cfg["deterministic"] = True
    if args.no_save:
        run_cfg["save"] = False
    if args.save_last:
        trainer_cfg["save_last"] = True
    if args.checkpoint_dir is not None:
        run_cfg["output_dir"] = args.checkpoint_dir

    if args.epochs is not None:
        trainer_cfg["epochs"] = args.epochs
    if args.lr is not None:
        optimizer_cfg["lr"] = args.lr
    if args.log_every is not None:
        trainer_cfg["log_every"] = args.log_every
    if args.eval_every is not None:
        trainer_cfg["eval_every"] = args.eval_every
    if args.save_every is not None:
        trainer_cfg["save_every"] = args.save_every
    if args.grad_clip_norm is not None:
        trainer_cfg["grad_clip_norm"] = args.grad_clip_norm
    if args.train_num_iterations is not None:
        trainer_cfg["train_num_iterations"] = args.train_num_iterations
    if args.eval_num_iterations is not None:
        trainer_cfg["eval_num_iterations"] = args.eval_num_iterations

    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.train_subset is not None:
        data_cfg["train_subset"] = args.train_subset
    if args.test_subset is not None:
        data_cfg["test_subset"] = args.test_subset

    return config


def _print_run_summary(config: dict[str, Any], *, dry_run: bool) -> None:
    model_name = config.get("model", {}).get("name")
    algorithm_name = config.get("algorithm", {}).get("name")
    dataset_name = config.get("data", {}).get("name")
    epochs = config.get("trainer", {}).get("epochs")
    batch_size = config.get("data", {}).get("config", {}).get("batch_size")
    output_dir = config.get("config", {}).get("output_dir")
    prefix = "dry_run" if dry_run else "run"
    print(
        f"{prefix}: algorithm={algorithm_name} model={model_name} "
        f"dataset={dataset_name} epochs={epochs} batch_size={batch_size} output_dir={output_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    resumed_config = None
    if args.resume_from is not None:
        resumed_config = _load_resume_config(args.resume_from)

    if resumed_config is not None:
        config = resumed_config
    else:
        config = load_experiment_config(
            config_name=args.config_name,
            config_dir=args.config_dir,
            algorithm=args.algorithm,
            trainer=args.trainer,
            data=args.data,
            model=args.model,
        )
    config = _apply_overrides(config, args)
    if args.dry_run:
        config.setdefault("config", {})["save"] = False

    if args.print_config:
        print(json.dumps(config, indent=2))

    _print_run_summary(config, dry_run=args.dry_run)
    if resumed_config is not None:
        print(f"resume: using config snapshot from {(Path(args.resume_from).expanduser().resolve().parent / 'experiment_config.json').resolve()}")

    model = build_model_from_config(config)
    trainer = build_trainer_from_config(model, config)

    if args.dry_run:
        print(f"dry_run: built {len(model.blocks)} block(s) on device={trainer.device}")
        if args.resume_from is not None:
            print(f"dry_run: would resume from {Path(args.resume_from).resolve()}")
        return 0

    if args.resume_from is not None:
        resume_path = Path(args.resume_from).expanduser().resolve()
        trainer.load_checkpoint(resume_path)
        if args.extra_epochs is not None:
            trainer.config.epochs = trainer.history.epochs_completed + args.extra_epochs
        print(
            f"resume: loaded {resume_path} "
            f"at epoch={trainer.history.epochs_completed} step={trainer.history.steps_completed}"
        )
    elif args.extra_epochs is not None:
        raise ValueError("--extra-epochs requires --resume-from.")

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
