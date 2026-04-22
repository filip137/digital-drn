from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from digital_drn.training.experiment import (
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    load_experiment_config,
)


CASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = CASE_DIR / "runs"
SUMMARY_PATH = CASE_DIR / "summary.json"

CONFIG_NAME = "cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc"
DEFAULT_V_OFF_WIDTHS = [0.5, 1.0, 1.2, 1.5]
DEFAULT_EPOCHS = 30
TRAIN_SUBSET = 128
TEST_SUBSET = 128
BATCH_SIZE = 16


def _parse_widths(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _prepare_config(width: float, *, device: str, epochs: int) -> dict:
    cfg = load_experiment_config(config_name=CONFIG_NAME)
    cfg = deepcopy(cfg)
    cfg["config"]["device"] = device
    cfg["config"]["output_dir"] = str(RUNS_DIR)
    cfg["trainer"]["epochs"] = int(epochs)
    cfg["data"]["config"]["train_subset"] = TRAIN_SUBSET
    cfg["data"]["config"]["test_subset"] = TEST_SUBSET
    cfg["data"]["config"]["batch_size"] = BATCH_SIZE
    cfg["data"]["config"]["num_workers"] = 0
    cfg["model"]["name"] = f"{cfg['model']['name']}_voff{str(width).replace('.', 'p')}"

    params = cfg["model"]["config"]["hard_sigmoid_param"]
    params["g_on"] = 10.0
    params["g_off"] = 1.0e-7
    params["v_min"] = -float(width)
    params["v_max"] = float(width)
    return cfg


def _run_one(width: float, *, device: str, epochs: int) -> dict:
    cfg = _prepare_config(width, device=device, epochs=epochs)
    start = time.time()
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    history = trainer.fit(train_loader, eval_loader)
    eval_metrics = trainer.evaluate(eval_loader) if eval_loader is not None else None
    row = {
        "v_off_width": float(width),
        "v_min": float(cfg["model"]["config"]["hard_sigmoid_param"]["v_min"]),
        "v_max": float(cfg["model"]["config"]["hard_sigmoid_param"]["v_max"]),
        "g_on": float(cfg["model"]["config"]["hard_sigmoid_param"]["g_on"]),
        "g_off": float(cfg["model"]["config"]["hard_sigmoid_param"]["g_off"]),
        "epochs": int(trainer.config.epochs),
        "train_final": float(history.train_accuracy[-1]),
        "train_best": float(max(history.train_accuracy)),
        "val_final": float(eval_metrics.accuracy if eval_metrics is not None else float("nan")),
        "val_best": float(max(history.val_accuracy) if history.val_accuracy else float("nan")),
        "checkpoint_dir": str(trainer.config.checkpoint_dir),
        "seconds": round(time.time() - start, 2),
    }
    trainer.close()
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[done] width={width:.2f} train_best={row['train_best']:.4f} "
        f"val_best={row['val_best']:.4f} seconds={row['seconds']:.2f}"
    )
    return row


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick v_off-width sweep for the widened CIFAR hard-sigmoid config.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--widths", default=",".join(str(v) for v in DEFAULT_V_OFF_WIDTHS))
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    widths = _parse_widths(args.widths)
    rows = [_run_one(width, device=args.device, epochs=args.epochs) for width in widths]
    rows.sort(key=lambda row: (-row["train_best"], -row["val_best"], row["seconds"]))
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2))
    print(f"[summary] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
