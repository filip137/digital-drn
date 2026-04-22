from __future__ import annotations

import argparse
import csv
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
SUMMARY_JSON = CASE_DIR / "summary.json"
SUMMARY_CSV = CASE_DIR / "summary.csv"
README_PATH = CASE_DIR / "README.md"

CONFIG_NAME = "cifar10_drn_only_signed_norm_readout_wider"
DEFAULT_EPOCHS = 100
TRAIN_SUBSET = 128
TEST_SUBSET = 128
BATCH_SIZE = 16

DIGITAL_MULTS = [1.0, 3.0, 10.0]
DRN_MULTS = [1.0, 3.0, 10.0]


def _slug(value: float) -> str:
    if abs(value - round(value)) <= 1.0e-12:
        return f"{int(round(value))}x"
    return f"{value:g}x".replace(".", "p")


def _prepare_config(*, device: str, epochs: int, digital_mult: float, drn_mult: float) -> tuple[dict, dict]:
    cfg = deepcopy(load_experiment_config(config_name=CONFIG_NAME))
    cfg.setdefault("algorithm", {})
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

    base = {
        "trainer_lr": float(cfg["trainer"]["optimizer"]["lr"]),
        "ff0_lr": float(cfg["model"]["blocks_config"][0]["ff"]["lr"]),
        "ff1_lr": float(cfg["model"]["blocks_config"][1]["ff"]["lr"]),
        "drn0_lr": float(cfg["model"]["blocks_config"][0]["drn"]["lr"]),
        "drn1_lr": float(cfg["model"]["blocks_config"][1]["drn"]["lr"]),
        "head_lr": float(cfg["model"]["head"]["lr"]),
    }

    cfg["trainer"]["optimizer"]["lr"] = base["trainer_lr"]
    cfg["model"]["blocks_config"][0]["ff"]["lr"] = base["ff0_lr"] * digital_mult
    cfg["model"]["blocks_config"][1]["ff"]["lr"] = base["ff1_lr"] * digital_mult
    cfg["model"]["blocks_config"][0]["drn"]["lr"] = base["drn0_lr"] * drn_mult
    cfg["model"]["blocks_config"][1]["drn"]["lr"] = base["drn1_lr"] * drn_mult
    cfg["model"]["head"]["lr"] = base["head_lr"] * digital_mult

    model_name = f"{cfg['model']['name']}_bp_overfit_dig{_slug(digital_mult)}_drn{_slug(drn_mult)}"
    cfg["model"]["name"] = model_name
    cfg.setdefault("_selection", {})
    cfg["_selection"]["model"] = model_name

    applied = {
        "digital_mult": float(digital_mult),
        "drn_mult": float(drn_mult),
        "trainer_lr": float(cfg["trainer"]["optimizer"]["lr"]),
        "ff_lr": float(cfg["model"]["blocks_config"][0]["ff"]["lr"]),
        "drn_lr": float(cfg["model"]["blocks_config"][0]["drn"]["lr"]),
        "head_lr": float(cfg["model"]["head"]["lr"]),
    }
    return cfg, applied


def _train_one(*, cfg: dict, applied: dict) -> dict:
    start = time.time()
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    history = trainer.fit(train_loader, eval_loader)
    checkpoint_dir = Path(trainer.config.checkpoint_dir) if trainer.config.checkpoint_dir is not None else None

    row = {
        **applied,
        "train_final": float(history.train_accuracy[-1]),
        "train_best": float(max(history.train_accuracy)),
        "val_final": float(history.val_accuracy[-1]) if history.val_accuracy else float("nan"),
        "val_best": float(max(history.val_accuracy)) if history.val_accuracy else float("nan"),
        "best_epoch_train": int(max(range(len(history.train_accuracy)), key=history.train_accuracy.__getitem__) + 1),
        "best_epoch_val": int(max(range(len(history.val_accuracy)), key=history.val_accuracy.__getitem__) + 1)
        if history.val_accuracy
        else None,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "seconds": round(time.time() - start, 2),
    }

    trainer.close()
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[run] dig={applied['digital_mult']:g}x drn={applied['drn_mult']:g}x "
        f"train_best={row['train_best']:.4f} val_best={row['val_best']:.4f} "
        f"seconds={row['seconds']:.2f}"
    )
    return row


def _write_reports(rows: list[dict], *, device: str, epochs: int) -> None:
    ranked = sorted(rows, key=lambda row: (-row["train_best"], -row["val_best"], row["seconds"]))
    summary = {
        "config_name": CONFIG_NAME,
        "device": device,
        "epochs": int(epochs),
        "train_subset": TRAIN_SUBSET,
        "test_subset": TEST_SUBSET,
        "batch_size": BATCH_SIZE,
        "digital_mults": DIGITAL_MULTS,
        "drn_mults": DRN_MULTS,
        "rows": ranked,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)

    lines = [
        "# Widened BP LR Grid Overfit",
        "",
        f"- config: `{CONFIG_NAME}`",
        f"- device: `{device}`",
        f"- epochs: `{epochs}`",
        f"- train_subset: `{TRAIN_SUBSET}`",
        f"- test_subset: `{TEST_SUBSET}`",
        f"- digital_mults: `{DIGITAL_MULTS}`",
        f"- drn_mults: `{DRN_MULTS}`",
        "",
        "## Ranked By Train Best",
        "",
        "| digital_mult | drn_mult | ff_lr | drn_lr | head_lr | train_best | val_best | train_epoch | val_epoch | run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['digital_mult']:g} | {row['drn_mult']:g} | {row['ff_lr']:.4g} | {row['drn_lr']:.4g} | {row['head_lr']:.4g} | "
            f"{row['train_best']:.4f} | {row['val_best']:.4f} | {row['best_epoch_train']} | {row['best_epoch_val']} | "
            f"`{Path(row['checkpoint_dir']).name if row['checkpoint_dir'] else 'none'}` |"
        )
    README_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search BP learning rates on the widened CIFAR overfit gate.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for digital_mult in DIGITAL_MULTS:
        for drn_mult in DRN_MULTS:
            cfg, applied = _prepare_config(
                device=args.device,
                epochs=args.epochs,
                digital_mult=float(digital_mult),
                drn_mult=float(drn_mult),
            )
            row = _train_one(cfg=cfg, applied=applied)
            rows.append(row)

    _write_reports(rows, device=args.device, epochs=args.epochs)
    print(f"[summary] wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
