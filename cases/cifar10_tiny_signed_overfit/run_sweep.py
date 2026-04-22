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


CASE_DIR = Path(__file__).resolve().parent
SUMMARY_JSON_PATH = CASE_DIR / "summary.json"
SUMMARY_CSV_PATH = CASE_DIR / "summary.csv"
OUTPUT_ROOT = "simulation_results/cifar10_tiny_signed_overfit_sweep"

NONLINEARITIES = ("perfect_diode", "linear", "hard_sigmoid")
VOLTAGE_AMPS = (1.0, 4.0)
BETAS = (1.0e-3, 1.0e-2)

HARD_SIGMOID_PARAM = {
    "g_on": 10.0,
    "g_off": 0.1,
    "v_min": -1.2,
    "v_max": 1.2,
}


def _variant_name(non_linearity: str, voltage_amp: float, beta: float) -> str:
    amp_token = str(voltage_amp).replace(".", "p")
    beta_token = f"{beta:.0e}".replace("+", "")
    return f"cifar10_drn_only_tiny_signed_{non_linearity}_va{amp_token}_beta{beta_token}"


def _base_config() -> dict:
    config = load_experiment_config(
        algorithm="ep",
        model="cifar10_drn_only_signed_norm_readout_tiny",
        data="cifar10",
    )
    config["trainer"]["epochs"] = 100
    config["trainer"]["log_every"] = 10
    config["trainer"]["save_best"] = False
    config["trainer"]["save_last"] = False
    config["trainer"]["save_events"] = False
    config["trainer"]["optimizer"]["config"]["weight_decay"] = 0.0

    data_config = config["data"]["config"]
    data_config["batch_size"] = 16
    data_config["train_subset"] = 128
    data_config["test_subset"] = 128

    run_config = config["config"]
    run_config["device"] = "cuda"
    run_config["save"] = True
    run_config["output_dir"] = OUTPUT_ROOT

    return config


def _configure_variant(non_linearity: str, voltage_amp: float, beta: float) -> dict:
    config = _base_config()
    name = _variant_name(non_linearity, voltage_amp, beta)

    config["_selection"]["model"] = name
    config["model"]["name"] = name
    config["algorithm"]["config"]["beta"] = float(beta)

    model_config = config["model"]["config"]
    model_config["drn_non_linearity"] = non_linearity
    model_config["voltage_amp"] = float(voltage_amp)
    model_config["current_amp"] = 1.0
    model_config["hard_sigmoid_param"] = dict(HARD_SIGMOID_PARAM if non_linearity == "hard_sigmoid" else {})

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
    with SUMMARY_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict] = []

    for non_linearity in NONLINEARITIES:
        for voltage_amp in VOLTAGE_AMPS:
            for beta in BETAS:
                variant_name = _variant_name(non_linearity, voltage_amp, beta)
                config = _configure_variant(non_linearity, voltage_amp, beta)
                print(
                    f"[start] {variant_name} "
                    f"non_linearity={non_linearity} voltage_amp={voltage_amp} beta={beta}"
                )
                started_at = time.time()
                model = trainer = train_loader = eval_loader = history = None
                try:
                    model = build_model_from_config(config)
                    trainer = build_trainer_from_config(model, config)
                    train_loader, eval_loader = build_dataloaders_from_config(config, download=False)
                    history = trainer.fit(train_loader, eval_loader)

                    row = {
                        "variant": variant_name,
                        "non_linearity": non_linearity,
                        "voltage_amp": voltage_amp,
                        "beta": beta,
                        "checkpoint_dir": str(trainer.config.checkpoint_dir),
                        "seconds": round(time.time() - started_at, 2),
                        **_history_metrics(history),
                    }
                    row["overfit_success"] = bool((row["train_best"] or 0.0) >= 0.99)
                    rows.append(row)
                    _write_summary(rows)
                    print(
                        f"[done] {variant_name} "
                        f"train_best={row['train_best']:.4f} val_best={row['val_best']:.4f} "
                        f"seconds={row['seconds']}"
                    )
                except Exception as exc:  # pragma: no cover - experiment harness
                    row = {
                        "variant": variant_name,
                        "non_linearity": non_linearity,
                        "voltage_amp": voltage_amp,
                        "beta": beta,
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
                        "overfit_success": False,
                        "error": repr(exc),
                    }
                    rows.append(row)
                    _write_summary(rows)
                    print(f"[error] {variant_name} {exc!r}")
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
