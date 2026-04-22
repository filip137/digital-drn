from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from .cli import _apply_overrides, _load_resume_config, _print_run_summary, make_parser as make_train_parser
from .trex_common import local_config_path, normalize_training_args
from ..training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    load_experiment_config,
)

DEFAULT_OUTPUT_DIR = "simulation_results/trex_simulations"
DEFAULT_NON_LINEARITY = "perfect_diode"


def _build_train_args(
    *,
    config_name: str,
    config_dir: str | None,
    training_args: list[str],
) -> argparse.Namespace:
    normalized = normalize_training_args(training_args)
    reserved = {"--config-name", "--config-dir", "--checkpoint-dir", "--no-save", "--dry-run"}
    duplicates = [arg for arg in normalized if arg in reserved]
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Pass {joined} to this CLI directly, not through the remainder args.")

    forwarded: list[str] = ["--config-name", config_name]
    if config_dir is not None:
        forwarded.extend(["--config-dir", config_dir])
    forwarded.extend(normalized)
    return make_train_parser().parse_args(forwarded)


def _force_perfect_diode(config: dict[str, Any], *, output_dir: str) -> dict[str, Any]:
    forced = json.loads(json.dumps(config))
    run_cfg = forced.setdefault("config", {})
    run_cfg["save"] = True
    run_cfg["output_dir"] = output_dir

    model_cfg = forced.setdefault("model", {})
    model_defaults = model_cfg.setdefault("config", {})
    model_defaults["drn_non_linearity"] = DEFAULT_NON_LINEARITY

    for block in model_cfg.get("blocks_config", []):
        if not isinstance(block, dict):
            continue
        drn_cfg = block.setdefault("drn", {})
        if isinstance(drn_cfg, dict):
            drn_cfg["drn_non_linearity"] = DEFAULT_NON_LINEARITY

    return forced


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-trex-perfect-diode-run",
        description="Run a config locally with perfect_diode DRNs and save under simulation_results/trex_simulations.",
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    local_config_path(args.config_name, args.config_dir)
    train_args = _build_train_args(
        config_name=args.config_name,
        config_dir=args.config_dir,
        training_args=list(args.training_args),
    )

    resumed_config = None
    if train_args.resume_from is not None:
        resumed_config = _load_resume_config(train_args.resume_from)

    if resumed_config is not None:
        config = resumed_config
    else:
        config = load_experiment_config(
            config_name=train_args.config_name,
            config_dir=train_args.config_dir,
            algorithm=train_args.algorithm,
            trainer=train_args.trainer,
            data=train_args.data,
            model=train_args.model,
        )

    config = _apply_overrides(config, train_args)
    config = _force_perfect_diode(config, output_dir=args.output_dir)
    if args.dry_run:
        config.setdefault("config", {})["save"] = False

    if train_args.print_config:
        print(json.dumps(config, indent=2))

    _print_run_summary(config, dry_run=args.dry_run)
    print(
        f"+ digital-drn-train --config-name {shlex.quote(args.config_name)} "
        f"--checkpoint-dir {shlex.quote(args.output_dir)} "
        f"[forced drn_non_linearity={DEFAULT_NON_LINEARITY}]"
    )
    if resumed_config is not None:
        resume_snapshot = Path(train_args.resume_from).expanduser().resolve().parent / "experiment_config.json"
        print(f"resume: using config snapshot from {resume_snapshot.resolve()}")

    model = build_model_from_config(config)
    trainer = build_trainer_from_config(model, config)

    if args.dry_run:
        print(f"dry_run: built {len(model.blocks)} block(s) on device={trainer.device}")
        if train_args.resume_from is not None:
            print(f"dry_run: would resume from {Path(train_args.resume_from).resolve()}")
        return 0

    if train_args.resume_from is not None:
        resume_path = Path(train_args.resume_from).expanduser().resolve()
        trainer.load_checkpoint(resume_path)
        if train_args.extra_epochs is not None:
            trainer.config.epochs = trainer.history.epochs_completed + train_args.extra_epochs
        print(
            f"resume: loaded {resume_path} "
            f"at epoch={trainer.history.epochs_completed} step={trainer.history.steps_completed}"
        )
    elif train_args.extra_epochs is not None:
        raise ValueError("--extra-epochs requires --resume-from.")

    train_loader, eval_loader = build_dataloaders_from_config(config, download=train_args.download)
    history = trainer.fit(train_loader, eval_loader, max_steps=train_args.max_steps)

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
