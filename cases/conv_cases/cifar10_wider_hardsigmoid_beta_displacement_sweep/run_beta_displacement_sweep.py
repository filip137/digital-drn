from __future__ import annotations

import argparse
import csv
import json
import math
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
    _path_ratios,
    _stats,
)
from run_contribution_checks import _summarize_variant, _variant_config  # noqa: E402
from digital_drn.utils.misc import resolve_device, set_seed  # noqa: E402

DEFAULT_CASE_DIR = Path("/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_beta_displacement_sweep")


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _write_csvs(case_dir: Path, summary: dict) -> None:
    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = [
            "beta",
            "num_iterations",
            "voltage_amp",
            "current_amp",
            "amp_ratio",
            "overall_cosine",
            "ff_flat_cosine",
            "drive_flat_cosine",
            "drn_flat_cosine",
            "drn_local_mean_cosine",
            "drn_weight_mean_cosine",
            "drn_bias_mean_cosine",
            "drn_ep_bp_norm_ratio",
            "positive_relative_disp",
            "negative_relative_disp",
            "mean_relative_disp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            disp = row["displacement"]
            writer.writerow({
                "beta": row["beta"],
                "num_iterations": row["num_iterations"],
                "voltage_amp": row["voltage_amp"],
                "current_amp": row["current_amp"],
                "amp_ratio": row["amp_ratio"],
                "overall_cosine": row["overall"]["overall_cosine"],
                "ff_flat_cosine": row["path_rows"]["ff_all"]["cosine"],
                "drive_flat_cosine": row["path_rows"]["drive_all"]["cosine"],
                "drn_flat_cosine": row["path_rows"]["drn_all"]["cosine"],
                "drn_local_mean_cosine": row["tensor_stats"]["drn_all"]["mean_cosine"],
                "drn_weight_mean_cosine": row["tensor_stats"]["drn_weights"]["mean_cosine"],
                "drn_bias_mean_cosine": row["tensor_stats"]["drn_biases"]["mean_cosine"],
                "drn_ep_bp_norm_ratio": row["path_rows"]["drn_all"]["ep_bp_norm_ratio"],
                "positive_relative_disp": disp["positive_relative_disp"],
                "negative_relative_disp": disp["negative_relative_disp"],
                "mean_relative_disp": disp["mean_relative_disp"],
            })

    with (results_dir / "tensor_details.csv").open("w", newline="") as f:
        fieldnames = [
            "beta",
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


def _first_change(rows: list[dict], *, rel_tol: float) -> dict | None:
    if not rows:
        return None
    base = float(rows[0]["displacement"]["mean_relative_disp"])
    for row in rows[1:]:
        value = float(row["displacement"]["mean_relative_disp"])
        if base <= 0.0:
            changed = abs(value - base) > rel_tol
        else:
            changed = abs(value - base) / base > rel_tol
        if changed:
            return {
                "reference_beta": rows[0]["beta"],
                "reference_mean_relative_disp": base,
                "first_changed_beta": row["beta"],
                "first_changed_mean_relative_disp": value,
                "relative_change": abs(value - base) / base if base > 0.0 else float("nan"),
                "rel_tol": rel_tol,
            }
    return None


def _write_readme(case_dir: Path, summary: dict) -> None:
    rows = summary["rows"]
    change_1pct = summary.get("first_displacement_change_1pct")
    change_10pct = summary.get("first_displacement_change_10pct")
    lines = [
        "# Hard-Sigmoid Beta Displacement Sweep",
        "",
        "This case checks beta sensitivity after fixing the iteration override in hybrid EP.",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- voltage_amp A: `{summary['voltage_amp']}`",
        f"- current_amp B: `{summary['current_amp']}`",
        f"- A/B: `{summary['amp_ratio']}`",
        f"- num_iterations: `{summary['num_iterations']}`",
        "- amp_gradient_compensation: `False`",
        f"- amplify_first_free_layer: `{summary['amplify_first_free_layer']}`",
        "",
        "## Why This Case Exists",
        "",
        "An earlier 128-iteration diagnostic was invalid because `num_iterations=128` was applied to ordinary BP and the EP free phase, but the `+beta` and `-beta` nudged phases still used the block-local training minimizer default of 6 iterations.",
        "That has been fixed by passing the override into `BlockEquilibriumProp` and temporarily applying it to the nudged training minimizer. The fixed 128-step run is slower, as expected, because both nudged phases now also run 128 relaxation iterations.",
        "",
        "## Beta Sweep",
        "",
        "| beta | disp mean | disp + | disp - | overall cos | flat DRN cos | local DRN cos | weight mean cos | bias mean cos | DRN EP/BP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        disp = row["displacement"]
        lines.append(
            f"| {row['beta']:.6g} | {disp['mean_relative_disp']:.6g} | "
            f"{disp['positive_relative_disp']:.6g} | {disp['negative_relative_disp']:.6g} | "
            f"{row['overall']['overall_cosine']:.4f} | "
            f"{row['path_rows']['drn_all']['cosine']:.4f} | "
            f"{row['tensor_stats']['drn_all']['mean_cosine']:.4f} | "
            f"{row['tensor_stats']['drn_weights']['mean_cosine']:.4f} | "
            f"{row['tensor_stats']['drn_biases']['mean_cosine']:.4f} | "
            f"{row['path_rows']['drn_all']['ep_bp_norm_ratio']:.4g} |"
        )
    lines.extend(["", "## Displacement Change Point", ""])
    if change_1pct is None:
        lines.append("Across this beta range, mean relative displacement never changed by more than 1% from the smallest-beta row.")
    else:
        lines.append(
            f"Mean relative displacement first changed by more than 1% at `beta={change_1pct['first_changed_beta']:.6g}` "
            f"relative to `beta={change_1pct['reference_beta']:.6g}`."
        )
    if change_10pct is None:
        lines.append("It also never changed by more than 10% from the smallest-beta row.")
    else:
        lines.append(
            f"Mean relative displacement first changed by more than 10% at `beta={change_10pct['first_changed_beta']:.6g}` "
            f"relative to `beta={change_10pct['reference_beta']:.6g}`."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- For the amplified `A=4, B=1` 128-step diagnostic, increasing beta mainly improves the finite-difference gradient direction rather than changing the pooled free-vs-nudged displacement over the tested range.",
        "- The direction improvement is concentrated in DRN weights. Bias directions are already near perfect once the iteration mismatch is fixed.",
        "- The remaining EP/BP norm mismatch follows the amplification staircase and should be handled separately from direction cosine.",
        "",
        "## Related Fixed Single-Point Cases",
        "",
        "- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_fixed`: `A=4, B=1, beta=1e-2`; local DRN cosine improved after the iteration fix but deeper block weights were still weak.",
        "- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta0p1_fixed`: `A=4, B=1, beta=0.1`; DRN directions became much cleaner.",
        "- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta1_fixed`: `A=4, B=1, beta=1`; DRN tensor directions were nearly perfect, while norm scaling remained amplified.",
        "- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_a1_b1_iter128_fixed`: `A=1, B=1, beta=1e-2`; unamplified 128-step directions were mostly healthy, with weakness mainly in block-1 weights.",
        "",
        "CSV outputs:",
        "",
        "- `results/summary.csv`",
        "- `results/tensor_details.csv`",
    ])
    (case_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep beta for fixed A/B and report displacement/cosine behavior.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--voltage-amp", type=float, default=4.0)
    parser.add_argument("--current-amp", type=float, default=1.0)
    parser.add_argument("--num-iterations", type=int, default=128)
    parser.add_argument("--betas", default="1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10")
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

    betas = _parse_floats(args.betas)
    rows = []
    tensor_rows_all = []
    for beta in betas:
        print(
            f"[run] beta={beta:g} A={args.voltage_amp:g} B={args.current_amp:g} "
            f"iters={args.num_iterations}",
            flush=True,
        )
        cfg = _variant_config(
            base_cfg,
            voltage_amp=float(args.voltage_amp),
            current_amp=float(args.current_amp),
            amplify_first_free_layer=bool(args.amplify_first_free_layer),
        )
        cfg.setdefault("algorithm", {}).setdefault("config", {})["ad_hoc_amp_gradient_scale"] = False
        result = _summarize_variant(
            cfg,
            args.checkpoint_path,
            beta=float(beta),
            device=device,
            amp_gradient_compensation=False,
            num_iterations=int(args.num_iterations),
        )
        tensor_rows = _collect_tensor_rows(result, voltage_amp=float(args.voltage_amp), current_amp=float(args.current_amp))
        for tensor_row in tensor_rows:
            tensor_row["beta"] = float(beta)
            tensor_row["num_iterations"] = int(args.num_iterations)
        tensor_rows_all.extend(tensor_rows)
        drn_rows = [row for row in tensor_rows if "/drn" in row["group"]]
        weight_rows = [row for row in drn_rows if row["kind"] == "weight"]
        bias_rows = [row for row in drn_rows if row["kind"] == "bias"]
        path_rows = _path_ratios(result)
        row = {
            "beta": float(beta),
            "num_iterations": int(args.num_iterations),
            "voltage_amp": float(args.voltage_amp),
            "current_amp": float(args.current_amp),
            "amp_ratio": float(args.voltage_amp / args.current_amp),
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
            f"disp={row['displacement']['mean_relative_disp']:.6g} "
            f"drn_flat={path_rows['drn_all']['cosine']:.4f} "
            f"local_drn={row['tensor_stats']['drn_all']['mean_cosine']:.4f} "
            f"w_cos={row['tensor_stats']['drn_weights']['mean_cosine']:.4f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "mode": "beta_displacement_sweep",
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(device),
        "voltage_amp": float(args.voltage_amp),
        "current_amp": float(args.current_amp),
        "amp_ratio": float(args.voltage_amp / args.current_amp),
        "num_iterations": int(args.num_iterations),
        "betas": betas,
        "amp_gradient_compensation": False,
        "amplify_first_free_layer": bool(args.amplify_first_free_layer),
        "rows": rows,
        "tensor_rows": tensor_rows_all,
    }
    summary["first_displacement_change_1pct"] = _first_change(rows, rel_tol=0.01)
    summary["first_displacement_change_10pct"] = _first_change(rows, rel_tol=0.10)
    results_dir = case_dir / "results"
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csvs(case_dir, summary)
    _write_readme(case_dir, summary)
    print(f"[summary] wrote {results_dir / 'summary.json'}", flush=True)
    print(f"[readme] wrote {case_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
