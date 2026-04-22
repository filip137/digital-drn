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
DEFAULT_EPOCHS = 100
DEFAULT_VOLTAGE_AMPS = [1.0, 2.0, 4.0, 8.0]
CURRENT_AMP = 1.0
DEFAULT_V_OFF_WIDTH = 1.0
DEFAULT_INIT_DRIVE_SCALE = 1.0
TRAIN_SUBSET = 128
TEST_SUBSET = 128
BATCH_SIZE = 16


def _parse_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _prepare_config(
    *,
    voltage_amp: float,
    v_off_width: float,
    init_drive_scale: float,
    device: str,
    epochs: int,
) -> dict:
    cfg = load_experiment_config(config_name=CONFIG_NAME)
    cfg = deepcopy(cfg)

    cfg["algorithm"] = {"name": "bp", "config": {}}

    cfg.setdefault("config", {})
    cfg["config"]["device"] = device
    cfg["config"]["output_dir"] = str(RUNS_DIR)

    cfg.setdefault("trainer", {})
    cfg["trainer"]["epochs"] = int(epochs)

    cfg.setdefault("data", {}).setdefault("config", {})
    data_cfg = cfg["data"]["config"]
    data_cfg["train_subset"] = TRAIN_SUBSET
    data_cfg["test_subset"] = TEST_SUBSET
    data_cfg["batch_size"] = BATCH_SIZE
    data_cfg["num_workers"] = 0

    cfg.setdefault("model", {})
    voff_tag = str(v_off_width).replace(".", "p")
    drive_tag = str(init_drive_scale).replace(".", "p")
    va_tag = str(voltage_amp).replace(".", "p")
    model_name = f"{cfg['model']['name']}_voff{voff_tag}_drive{drive_tag}_va{va_tag}_bp"
    cfg["model"]["name"] = model_name
    cfg.setdefault("_selection", {})
    cfg["_selection"]["model"] = model_name

    model_cfg = cfg["model"]["config"]
    model_cfg["drn_non_linearity"] = "hard_sigmoid"
    model_cfg["voltage_amp"] = float(voltage_amp)
    model_cfg["current_amp"] = float(CURRENT_AMP)

    model_cfg["init_drive_scale"] = float(init_drive_scale)
    params = model_cfg["hard_sigmoid_param"]
    params["g_on"] = 10.0
    params["g_off"] = 1.0e-7
    params["v_min"] = -float(v_off_width)
    params["v_max"] = float(v_off_width)
    return cfg


def _run_one(
    *,
    voltage_amp: float,
    v_off_width: float,
    init_drive_scale: float,
    device: str,
    epochs: int,
) -> dict:
    cfg = _prepare_config(
        voltage_amp=voltage_amp,
        v_off_width=v_off_width,
        init_drive_scale=init_drive_scale,
        device=device,
        epochs=epochs,
    )
    start = time.time()
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    history = trainer.fit(train_loader, eval_loader)
    eval_metrics = trainer.evaluate(eval_loader) if eval_loader is not None else None
    checkpoint_dir = Path(trainer.config.checkpoint_dir) if trainer.config.checkpoint_dir is not None else None

    row = {
        "algorithm": "bp",
        "voltage_amp": float(voltage_amp),
        "current_amp": float(CURRENT_AMP),
        "init_drive_scale": float(init_drive_scale),
        "hard_sigmoid": {
            "g_on": 10.0,
            "g_off": 1.0e-7,
            "v_min": -float(v_off_width),
            "v_max": float(v_off_width),
        },
        "epochs": int(trainer.config.epochs),
        "train_final": float(history.train_accuracy[-1]),
        "train_best": float(max(history.train_accuracy)),
        "val_final": float(eval_metrics.accuracy if eval_metrics is not None else float("nan")),
        "val_best": float(max(history.val_accuracy) if history.val_accuracy else float("nan")),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "seconds": round(time.time() - start, 2),
    }
    trainer.close()
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[done] va={voltage_amp:g} voff={v_off_width:g} drive={init_drive_scale:g} "
        f"train_best={row['train_best']:.4f} "
        f"val_best={row['val_best']:.4f} seconds={row['seconds']:.2f}"
    )
    return row


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BP-only voltage_amp sweep for the widened hard-sigmoid CIFAR config."
    )
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--voltage-amps", default=",".join(str(v) for v in DEFAULT_VOLTAGE_AMPS))
    parser.add_argument("--v-off-width", type=float, default=DEFAULT_V_OFF_WIDTH)
    parser.add_argument("--init-drive-scale", type=float, default=DEFAULT_INIT_DRIVE_SCALE)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    voltage_amps = _parse_values(args.voltage_amps)
    rows = [
        _run_one(
            voltage_amp=va,
            v_off_width=float(args.v_off_width),
            init_drive_scale=float(args.init_drive_scale),
            device=args.device,
            epochs=args.epochs,
        )
        for va in voltage_amps
    ]
    rows.sort(key=lambda row: (-row["train_best"], -row["val_best"], row["seconds"]))
    summary = {
        "config_name": CONFIG_NAME,
        "device": args.device,
        "epochs": int(args.epochs),
        "v_off_width": float(args.v_off_width),
        "init_drive_scale": float(args.init_drive_scale),
        "rows": rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
