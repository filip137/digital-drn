from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

from digital_drn.training.experiment import build_dataloaders_from_config, build_model_from_config  # noqa: E402
from digital_drn.training.loaded_model_beta_sweep import _load_checkpoint_into_model, _load_json  # noqa: E402
from digital_drn.utils.misc import resolve_device, set_seed  # noqa: E402


def _update_stats(stats: dict[str, float], tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float()
    abs_tensor = tensor.abs()
    stats["sum_sq"] += float(torch.sum(tensor * tensor).item())
    stats["sum_abs"] += float(torch.sum(abs_tensor).item())
    stats["numel"] += int(tensor.numel())
    stats["max_abs"] = max(stats["max_abs"], float(abs_tensor.max().item()))


def _empty_stats() -> dict[str, float]:
    return {
        "sum_sq": 0.0,
        "sum_abs": 0.0,
        "numel": 0.0,
        "max_abs": 0.0,
    }


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    numel = float(stats["numel"])
    if numel <= 0:
        return {
            "rms": float("nan"),
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "numel": 0,
        }
    return {
        "rms": math.sqrt(stats["sum_sq"] / numel),
        "mean_abs": stats["sum_abs"] / numel,
        "max_abs": stats["max_abs"],
        "numel": int(numel),
    }


def _collect_residual_currents(model, loader, device: torch.device) -> dict:
    layer_stats: OrderedDict[str, dict[str, float]] = OrderedDict()
    block_stats: OrderedDict[str, dict[str, float]] = OrderedDict()
    overall = _empty_stats()

    model.eval()

    for batch_inputs, _batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        _ = model(batch_inputs, reset=True)
        for block_idx, block in enumerate(model.blocks):
            block_key = f"block_{block_idx}"
            block_acc = block_stats.setdefault(block_key, _empty_stats())
            for layer_idx, layer in enumerate(block.free_layers()):
                layer_key = f"{block_key}/layer_{layer_idx}"
                layer_acc = layer_stats.setdefault(layer_key, _empty_stats())
                residual = block.energy.grad_layer_fn(layer)()
                _update_stats(layer_acc, residual)
                _update_stats(block_acc, residual)
                _update_stats(overall, residual)
        model.detach_state_()

    finalized_layers = OrderedDict((name, _finalize_stats(stats)) for name, stats in layer_stats.items())
    finalized_blocks = OrderedDict((name, _finalize_stats(stats)) for name, stats in block_stats.items())
    finalized_overall = _finalize_stats(overall)

    worst_layer_by_max = max(finalized_layers.items(), key=lambda item: item[1]["max_abs"])
    worst_layer_by_rms = max(finalized_layers.items(), key=lambda item: item[1]["rms"])

    return {
        "overall": finalized_overall,
        "blocks": finalized_blocks,
        "layers": finalized_layers,
        "worst_layer_by_max_abs": {
            "layer": worst_layer_by_max[0],
            **worst_layer_by_max[1],
        },
        "worst_layer_by_rms": {
            "layer": worst_layer_by_rms[0],
            **worst_layer_by_rms[1],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze free-phase residual currents for the hard-sigmoid va sweep.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(__file__).resolve().with_name("summary.json"),
        help="Path to the completed sweep summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().with_name("residual_currents_summary.json"),
        help="Path to write the residual-current summary JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device override for the residual-current analysis",
    )
    parser.add_argument(
        "--use-train-loader",
        action="store_true",
        help="Analyze the training slice instead of the eval slice",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the residual-current analysis loader",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary_path.read_text())
    rows = list(summary.get("rows", []))
    if not rows:
        raise ValueError(f"No rows found in {args.summary_path}.")

    output_rows = []
    for row in rows:
        run_dir = Path(row["checkpoint_dir"])
        config = _load_json(run_dir / "experiment_config.json")
        set_seed(int(config.get("config", {}).get("seed", 0)))
        model = build_model_from_config(config)
        _load_checkpoint_into_model(model, run_dir / "checkpoint_best.pt")

        requested_device = args.device or config.get("config", {}).get("device", "auto")
        device = resolve_device(requested_device)
        model.to(device)

        train_loader, eval_loader = build_dataloaders_from_config(config, download=False)
        base_loader = train_loader if args.use_train_loader else eval_loader
        loader = DataLoader(
            base_loader.dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        residuals = _collect_residual_currents(model, loader, device)
        output_rows.append(
            {
                "algorithm": row.get("algorithm"),
                "voltage_amp": row.get("voltage_amp"),
                "current_amp": row.get("current_amp"),
                "init_drive_scale": row.get("init_drive_scale"),
                "hard_sigmoid": dict(row.get("hard_sigmoid", {})),
                "epochs": row.get("epochs"),
                "checkpoint_dir": str(run_dir),
                "dataset_slice": "train_loader" if args.use_train_loader else "eval_loader",
                "residual_currents": residuals,
            }
        )

    output_rows.sort(key=lambda row: row["residual_currents"]["overall"]["max_abs"])
    result = {
        "summary_path": str(args.summary_path),
        "device": args.device,
        "dataset_slice": "train_loader" if args.use_train_loader else "eval_loader",
        "rows": output_rows,
    }
    args.output_path.write_text(json.dumps(result, indent=2))
    print(f"[summary] wrote {args.output_path}")


if __name__ == "__main__":
    main()
