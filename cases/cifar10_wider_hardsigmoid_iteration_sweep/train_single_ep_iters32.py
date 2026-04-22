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
RUNS_DIR = CASE_DIR / "single_ep_iters32_runs"
SUMMARY_PATH = CASE_DIR / "single_ep_iters32_summary.json"

CONFIG_NAME = "cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc"
DEFAULT_EPOCHS = 100
TRAIN_SUBSET = 128
TEST_SUBSET = 128
BATCH_SIZE = 16
NUM_ITERATIONS = 32
V_OFF_WIDTH = 1.0


def _prepare_config(*, device: str, epochs: int) -> dict:
    cfg = load_experiment_config(config_name=CONFIG_NAME)
    cfg = deepcopy(cfg)

    cfg.setdefault("config", {})
    cfg["config"]["device"] = device
    cfg["config"]["output_dir"] = str(RUNS_DIR)

    cfg.setdefault("trainer", {})
    cfg["trainer"]["epochs"] = int(epochs)
    cfg["trainer"]["train_num_iterations"] = NUM_ITERATIONS
    cfg["trainer"]["eval_num_iterations"] = NUM_ITERATIONS

    cfg.setdefault("data", {}).setdefault("config", {})
    data_cfg = cfg["data"]["config"]
    data_cfg["train_subset"] = TRAIN_SUBSET
    data_cfg["test_subset"] = TEST_SUBSET
    data_cfg["batch_size"] = BATCH_SIZE
    data_cfg["num_workers"] = 0

    cfg.setdefault("algorithm", {})
    cfg["algorithm"]["name"] = "ep"

    cfg.setdefault("model", {})
    cfg["model"]["name"] = f"{cfg['model']['name']}_voff1p0_ep_iters32"
    cfg.setdefault("_selection", {})
    cfg["_selection"]["model"] = cfg["model"]["name"]

    params = cfg["model"]["config"]["hard_sigmoid_param"]
    params["g_on"] = 10.0
    params["g_off"] = 1.0e-7
    params["v_min"] = -V_OFF_WIDTH
    params["v_max"] = V_OFF_WIDTH

    for block in cfg["model"]["blocks_config"]:
        block.setdefault("drn", {})
        block["drn"]["num_iterations"] = NUM_ITERATIONS

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a single EP hard-sigmoid widened run with 32 solver iterations.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _prepare_config(device=args.device, epochs=args.epochs)
    start = time.time()

    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    history = trainer.fit(train_loader, eval_loader)
    eval_metrics = trainer.evaluate(eval_loader) if eval_loader is not None else None
    checkpoint_dir = Path(trainer.config.checkpoint_dir) if trainer.config.checkpoint_dir is not None else None
    checkpoint_best = checkpoint_dir / "checkpoint_best.pt" if checkpoint_dir is not None else None

    summary = {
        "config_name": CONFIG_NAME,
        "device": args.device,
        "epochs": int(trainer.config.epochs),
        "num_iterations": NUM_ITERATIONS,
        "v_off_width": V_OFF_WIDTH,
        "hard_sigmoid": {
            "g_on": 10.0,
            "g_off": 1.0e-7,
            "v_min": -V_OFF_WIDTH,
            "v_max": V_OFF_WIDTH,
        },
        "row": {
            "algorithm": "ep",
            "train_final": float(history.train_accuracy[-1]),
            "train_best": float(max(history.train_accuracy)),
            "val_final": float(eval_metrics.accuracy if eval_metrics is not None else float("nan")),
            "val_best": float(max(history.val_accuracy) if history.val_accuracy else float("nan")),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "checkpoint_best": str(checkpoint_best) if checkpoint_best is not None else None,
            "seconds": round(time.time() - start, 2),
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(
        f"[train] train_best={summary['row']['train_best']:.4f} "
        f"val_best={summary['row']['val_best']:.4f} "
        f"seconds={summary['row']['seconds']:.2f}"
    )
    print(f"[summary] wrote {SUMMARY_PATH}")

    trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
