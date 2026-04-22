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
from digital_drn.training.loaded_model_beta_sweep import (
    _aggregate_sweep,
    _load_checkpoint_into_model,
)
from digital_drn.training.trainer import build_criterion


CASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = CASE_DIR / "runs"
SUMMARY_PATH = CASE_DIR / "summary.json"

CONFIG_NAME = "cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc"
DEFAULT_EPOCHS = 100
DEFAULT_BETA = 1.0e-2
V_OFF_WIDTH = 1.0
TRAIN_SUBSET = 128
TEST_SUBSET = 128
BATCH_SIZE = 16


def _prepare_config(*, algorithm_name: str, device: str, epochs: int) -> dict:
    cfg = load_experiment_config(config_name=CONFIG_NAME)
    cfg = deepcopy(cfg)

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
    model_name = f"{cfg['model']['name']}_voff1p0_{algorithm_name}"
    cfg["model"]["name"] = model_name
    cfg.setdefault("_selection", {})
    cfg["_selection"]["model"] = model_name

    params = cfg["model"]["config"]["hard_sigmoid_param"]
    params["g_on"] = 10.0
    params["g_off"] = 1.0e-7
    params["v_min"] = -V_OFF_WIDTH
    params["v_max"] = V_OFF_WIDTH

    if algorithm_name == "bp":
        cfg["algorithm"] = {"name": "bp", "config": {}}
    elif algorithm_name == "ep":
        cfg.setdefault("algorithm", {})
        cfg["algorithm"]["name"] = "ep"
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm_name}'.")
    return cfg


def _train_one(*, algorithm_name: str, device: str, epochs: int) -> dict:
    cfg = _prepare_config(algorithm_name=algorithm_name, device=device, epochs=epochs)
    start = time.time()
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    history = trainer.fit(train_loader, eval_loader)
    eval_metrics = trainer.evaluate(eval_loader) if eval_loader is not None else None
    checkpoint_dir = Path(trainer.config.checkpoint_dir) if trainer.config.checkpoint_dir is not None else None
    checkpoint_best = checkpoint_dir / "checkpoint_best.pt" if checkpoint_dir is not None else None

    row = {
        "algorithm": algorithm_name,
        "epochs": int(trainer.config.epochs),
        "train_final": float(history.train_accuracy[-1]),
        "train_best": float(max(history.train_accuracy)),
        "val_final": float(eval_metrics.accuracy if eval_metrics is not None else float("nan")),
        "val_best": float(max(history.val_accuracy) if history.val_accuracy else float("nan")),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_best": str(checkpoint_best) if checkpoint_best is not None else None,
        "config": cfg,
        "seconds": round(time.time() - start, 2),
    }
    trainer.close()
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[train] algorithm={algorithm_name} "
        f"train_best={row['train_best']:.4f} val_best={row['val_best']:.4f} "
        f"seconds={row['seconds']:.2f}"
    )
    return row


def _diagnose_one(row: dict, *, beta: float, device: str) -> dict:
    cfg = deepcopy(row["config"])
    model = build_model_from_config(cfg)
    resolved_device = torch.device(device)
    model.set_device(resolved_device)
    _load_checkpoint_into_model(model, Path(row["checkpoint_best"]))
    model.enable_resistive_grad_()
    model.eval()
    model.detach_state_()

    _train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    criterion = build_criterion(cfg.get("trainer", {}).get("criterion", "cross_entropy")).to(resolved_device)
    diag_rows = _aggregate_sweep(
        model,
        eval_loader,
        criterion,
        [float(beta)],
        num_iterations=None,
        amp_gradient_compensation=bool(cfg.get("algorithm", {}).get("config", {}).get("ad_hoc_amp_gradient_scale", False)),
    )
    diagnosis = diag_rows[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnosis


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the widened hard-sigmoid CIFAR config for 100 epochs under EP and BP, then diagnose BP-vs-EP agreement."
    )
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "config_name": CONFIG_NAME,
        "device": args.device,
        "epochs": int(args.epochs),
        "v_off_width": V_OFF_WIDTH,
        "hard_sigmoid": {
            "g_on": 10.0,
            "g_off": 1.0e-7,
            "v_min": -V_OFF_WIDTH,
            "v_max": V_OFF_WIDTH,
        },
        "diagnostic_beta": float(args.beta),
        "rows": [],
    }

    for algorithm_name in ("ep", "bp"):
        row = _train_one(algorithm_name=algorithm_name, device=args.device, epochs=args.epochs)
        diagnosis = _diagnose_one(row, beta=float(args.beta), device=args.device)
        row["diagnosis"] = diagnosis
        row.pop("config", None)
        summary["rows"].append(row)
        print(
            f"[diagnose] algorithm={algorithm_name} "
            f"overall_cosine={diagnosis['overall_cosine']:.4f} "
            f"disp={diagnosis['displacement']['mean_relative_disp']:.4f}"
        )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
