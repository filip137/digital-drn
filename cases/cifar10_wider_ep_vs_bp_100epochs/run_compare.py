from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import time

import torch

from digital_drn.training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    load_experiment_config,
)
from digital_drn.training.loaded_model_beta_sweep import _aggregate_sweep
from digital_drn.training.trainer import build_criterion


CASE_DIR = Path(__file__).resolve().parent
SUMMARY_JSON_PATH = CASE_DIR / "summary.json"
SUMMARY_CSV_PATH = CASE_DIR / "summary.csv"
OUTPUT_ROOT = CASE_DIR / "runs"

ALGORITHMS = ("ep", "bp")


def _configure_algorithm(algorithm: str) -> dict:
    config = load_experiment_config(
        config_name="cifar10_drn_only_signed_norm_readout_wider",
        algorithm=algorithm,
    )

    trainer_cfg = config["trainer"]
    trainer_cfg["epochs"] = 100
    trainer_cfg["log_every"] = 10
    trainer_cfg["save_best"] = False
    trainer_cfg["save_last"] = False
    trainer_cfg["save_events"] = False

    data_cfg = config["data"]["config"]
    data_cfg["batch_size"] = 16
    data_cfg["num_workers"] = 0
    data_cfg["train_subset"] = 128
    data_cfg["test_subset"] = 128

    run_cfg = config["config"]
    run_cfg["device"] = "cuda"
    run_cfg["save"] = True
    run_cfg["output_dir"] = str(OUTPUT_ROOT)

    config["_selection"]["model"] = f"cifar10_wider_{algorithm}"
    config["model"]["name"] = "cifar10_drn_only_signed_norm_readout_wider"
    return config


def _history_metrics(history) -> dict[str, float | int | None]:
    train_accuracy = list(history.train_accuracy)
    val_accuracy = list(history.val_accuracy)
    train_loss = list(history.train_loss)
    val_loss = list(history.val_loss)

    best_train = max(train_accuracy) if train_accuracy else None
    best_val = max(val_accuracy) if val_accuracy else None
    return {
        "epochs_recorded": len(train_accuracy),
        "train_final": train_accuracy[-1] if train_accuracy else None,
        "train_best": best_train,
        "train_best_epoch": (train_accuracy.index(best_train) + 1) if train_accuracy else None,
        "val_final": val_accuracy[-1] if val_accuracy else None,
        "val_best": best_val,
        "val_best_epoch": (val_accuracy.index(best_val) + 1) if val_accuracy else None,
        "train_loss_final": train_loss[-1] if train_loss else None,
        "val_loss_final": val_loss[-1] if val_loss else None,
    }


def _write_summary(rows: list[dict]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict] = []

    for algorithm in ALGORITHMS:
        print(f"[start] algorithm={algorithm}")
        started_at = time.time()
        model = trainer = train_loader = eval_loader = history = None
        try:
            config = _configure_algorithm(algorithm)
            model = build_model_from_config(config)
            trainer = build_trainer_from_config(model, config)
            train_loader, eval_loader = build_dataloaders_from_config(config, download=False)
            history = trainer.fit(train_loader, eval_loader)

            criterion = build_criterion(trainer.config.criterion).to(trainer.device)
            sweep_rows = _aggregate_sweep(
                model,
                eval_loader,
                criterion,
                [1.0e-2],
                num_iterations=trainer.config.eval_num_iterations,
            )
            diagnostic = sweep_rows[0]
            displacement = diagnostic["displacement"]

            row = {
                "algorithm": algorithm,
                "checkpoint_dir": str(trainer.config.checkpoint_dir),
                "seconds": round(time.time() - started_at, 2),
                **_history_metrics(history),
                "diagnostic_beta": 1.0e-2,
                "displacement_mean": displacement["mean_relative_disp"],
                "displacement_block0_mean": displacement["blocks"][0]["mean_relative_disp"],
                "displacement_block1_mean": displacement["blocks"][1]["mean_relative_disp"],
                "overall_cosine": diagnostic["overall_cosine"],
                "overall_relative_error": diagnostic["overall_relative_error"],
            }
            rows.append(row)
            _write_summary(rows)
            print(
                f"[done] algorithm={algorithm} "
                f"train_best={row['train_best']:.4f} "
                f"val_best={row['val_best']:.4f} "
                f"overall_cos={row['overall_cosine']:.4f} "
                f"disp={row['displacement_mean']:.4e}"
            )
        except Exception as exc:  # pragma: no cover - experiment harness
            row = {
                "algorithm": algorithm,
                "checkpoint_dir": None,
                "seconds": round(time.time() - started_at, 2),
                "epochs_recorded": None,
                "train_final": None,
                "train_best": None,
                "train_best_epoch": None,
                "val_final": None,
                "val_best": None,
                "val_best_epoch": None,
                "train_loss_final": None,
                "val_loss_final": None,
                "diagnostic_beta": 1.0e-2,
                "displacement_mean": None,
                "displacement_block0_mean": None,
                "displacement_block1_mean": None,
                "overall_cosine": None,
                "overall_relative_error": None,
                "error": repr(exc),
            }
            rows.append(row)
            _write_summary(rows)
            print(f"[error] algorithm={algorithm} {exc!r}")
        finally:
            if trainer is not None:
                trainer.close()
            del history, train_loader, eval_loader, trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
