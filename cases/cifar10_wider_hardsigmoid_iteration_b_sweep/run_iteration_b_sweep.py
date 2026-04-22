from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))
CONTRIB_CASE_DIR = REPO_ROOT / "cases" / "cifar10_wider_hardsigmoid_contribution_breakdown"
AMP_CASE_DIR = REPO_ROOT / "cases" / "cifar10_wider_hardsigmoid_amp_grid_scaling"
for path in (CONTRIB_CASE_DIR, AMP_CASE_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from run_amp_grid import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    _collect_tensor_rows,
    _parse_floats,
    _path_ratios,
    _stats,
)
from run_contribution_checks import _summarize_variant, _variant_config  # noqa: E402
from digital_drn.utils.misc import resolve_device, set_seed  # noqa: E402

DEFAULT_CASE_DIR = Path("/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_iteration_b_sweep")


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _write_csvs(case_dir: Path, summary: dict) -> None:
    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = [
            "num_iterations",
            "voltage_amp",
            "current_amp",
            "amp_ratio",
            "beta",
            "overall_cosine",
            "ff_flat_cosine",
            "drive_flat_cosine",
            "drn_flat_cosine",
            "drn_local_mean_cosine",
            "drn_weight_mean_cosine",
            "drn_bias_mean_cosine",
            "drn_ep_bp_norm_ratio",
            "mean_relative_disp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({
                "num_iterations": row["num_iterations"],
                "voltage_amp": row["voltage_amp"],
                "current_amp": row["current_amp"],
                "amp_ratio": row["amp_ratio"],
                "beta": row["beta"],
                "overall_cosine": row["overall"]["overall_cosine"],
                "ff_flat_cosine": row["path_rows"]["ff_all"]["cosine"],
                "drive_flat_cosine": row["path_rows"]["drive_all"]["cosine"],
                "drn_flat_cosine": row["path_rows"]["drn_all"]["cosine"],
                "drn_local_mean_cosine": row["tensor_stats"]["drn_all"]["mean_cosine"],
                "drn_weight_mean_cosine": row["tensor_stats"]["drn_weights"]["mean_cosine"],
                "drn_bias_mean_cosine": row["tensor_stats"]["drn_biases"]["mean_cosine"],
                "drn_ep_bp_norm_ratio": row["path_rows"]["drn_all"]["ep_bp_norm_ratio"],
                "mean_relative_disp": row["displacement"]["mean_relative_disp"],
            })

    with (results_dir / "tensor_details.csv").open("w", newline="") as f:
        fieldnames = [
            "num_iterations",
            "voltage_amp",
            "current_amp",
            "amp_ratio",
            "group",
            "tensor",
            "kind",
            "cosine",
            "relative_error",
            "bp_norm",
            "ep_norm",
            "ep_bp_norm_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["tensor_rows"])


def _write_readme(case_dir: Path, summary: dict) -> None:
    lines = [
        "# Hard-Sigmoid Iteration/B Cosine Sweep",
        "",
        "Uses the trained wider hard-sigmoid checkpoint and varies only solver iterations and current_amp=B.",
        "",
        f"- checkpoint: `{summary["checkpoint_path"]}`",
        f"- voltage_amp A: `{summary["voltage_amp"]}`",
        f"- beta: `{summary["beta"]}`",
        "- amp_gradient_compensation: `False`",
        f"- amplify_first_free_layer: `{summary["amplify_first_free_layer"]}`",
        "",
        "## Summary",
        "",
        "| iters | B | A/B | overall cos | ff cos | drive cos | flat DRN cos | local DRN cos | weight mean cos | bias mean cos | DRN EP/BP | disp |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row["num_iterations"]} | {row["current_amp"]:.4g} | {row["amp_ratio"]:.4g} | "
            f"{row["overall"]["overall_cosine"]:.4f} | "
            f"{row["path_rows"]["ff_all"]["cosine"]:.4f} | "
            f"{row["path_rows"]["drive_all"]["cosine"]:.4f} | "
            f"{row["path_rows"]["drn_all"]["cosine"]:.4f} | "
            f"{row["tensor_stats"]["drn_all"]["mean_cosine"]:.4f} | "
            f"{row["tensor_stats"]["drn_weights"]["mean_cosine"]:.4f} | "
            f"{row["tensor_stats"]["drn_biases"]["mean_cosine"]:.4f} | "
            f"{row["path_rows"]["drn_all"]["ep_bp_norm_ratio"]:.4g} | "
            f"{row["displacement"]["mean_relative_disp"]:.4g} |"
        )
    lines.extend([
        "",
        "CSV outputs:",
        "",
        "- `results/summary.csv`",
        "- `results/tensor_details.csv`",
    ])
    (case_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep solver iterations and B for BP-vs-EP cosine diagnostics.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--voltage-amp", type=float, default=4.0)
    parser.add_argument("--current-amps", default="0.25,0.5,1,2")
    parser.add_argument("--num-iterations", default="8,16,32,128")
    parser.add_argument("--beta", type=float, default=1.0e-2)
    parser.add_argument("--no-amplify-first-free-layer", dest="amplify_first_free_layer", action="store_false")
    parser.set_defaults(amplify_first_free_layer=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    base_cfg = json.loads(args.config_path.read_text())
    set_seed(int(base_cfg.get("config", {}).get("seed", 0)))
    device = resolve_device(args.device)
    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "results").mkdir(parents=True, exist_ok=True)

    current_amps = _parse_floats(args.current_amps)
    num_iterations_values = _parse_ints(args.num_iterations)

    rows = []
    tensor_rows_all = []
    for num_iterations in num_iterations_values:
        for current_amp in current_amps:
            print(f"[run] iters={num_iterations} A={args.voltage_amp:g} B={current_amp:g} ratio={args.voltage_amp/current_amp:g}")
            cfg = _variant_config(
                base_cfg,
                voltage_amp=float(args.voltage_amp),
                current_amp=float(current_amp),
                amplify_first_free_layer=bool(args.amplify_first_free_layer),
            )
            cfg.setdefault("algorithm", {}).setdefault("config", {})["ad_hoc_amp_gradient_scale"] = False
            result = _summarize_variant(
                cfg,
                args.checkpoint_path,
                beta=float(args.beta),
                device=device,
                amp_gradient_compensation=False,
                num_iterations=int(num_iterations),
            )
            tensor_rows = _collect_tensor_rows(result, voltage_amp=float(args.voltage_amp), current_amp=float(current_amp))
            for tensor_row in tensor_rows:
                tensor_row["num_iterations"] = int(num_iterations)
            tensor_rows_all.extend(tensor_rows)
            drn_rows = [row for row in tensor_rows if "/drn" in row["group"]]
            weight_rows = [row for row in drn_rows if row["kind"] == "weight"]
            bias_rows = [row for row in drn_rows if row["kind"] == "bias"]
            path_rows = _path_ratios(result)
            row = {
                "num_iterations": int(num_iterations),
                "voltage_amp": float(args.voltage_amp),
                "current_amp": float(current_amp),
                "amp_ratio": float(args.voltage_amp / current_amp),
                "beta": float(args.beta),
                "overall": result["overall"],
                "path_rows": path_rows,
                "group_rows": result["group_rows"],
                "tensor_stats": {
                    "drn_all": _stats(drn_rows),
                    "drn_weights": _stats(weight_rows),
                    "drn_biases": _stats(bias_rows),
                },
                "displacement": result["displacement"],
            }
            rows.append(row)
            print(
                "  "
                f"overall={row["overall"]["overall_cosine"]:.4f} "
                f"drn_flat={path_rows["drn_all"]["cosine"]:.4f} "
                f"local_drn={row["tensor_stats"]["drn_all"]["mean_cosine"]:.4f} "
                f"drn_epbp={path_rows["drn_all"]["ep_bp_norm_ratio"]:.4g}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "mode": "iteration_b_sweep",
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(device),
        "voltage_amp": float(args.voltage_amp),
        "current_amps": current_amps,
        "num_iterations_values": num_iterations_values,
        "beta": float(args.beta),
        "amp_gradient_compensation": False,
        "amplify_first_free_layer": bool(args.amplify_first_free_layer),
        "rows": rows,
        "tensor_rows": tensor_rows_all,
    }
    results_dir = case_dir / "results"
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csvs(case_dir, summary)
    _write_readme(case_dir, summary)
    print(f"[summary] wrote {results_dir / "summary.json"}")
    print(f"[readme] wrote {case_dir / "README.md"}")


if __name__ == "__main__":
    main()
