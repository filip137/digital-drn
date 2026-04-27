from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch

from digital_drn.core.parameter import Bias
from digital_drn.training.experiment import build_dataloaders_from_config, build_model_from_config
from digital_drn.training.loaded_model_beta_sweep import (
    _accumulate_group_sums,
    _analyze_batch,
    _cosine,
    _finalize_group_means,
    _flatten,
    _group_metrics,
    _group_template,
    _load_checkpoint_into_model,
    _overall_metrics,
)
from digital_drn.training.trainer import build_criterion
from digital_drn.utils.misc import resolve_device, set_seed


DEFAULT_CONFIG_PATH = Path(
    "/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/"
    "runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/"
    "20260408-163101-nom-cool-2/experiment_config.json"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/"
    "runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/"
    "20260408-163101-nom-cool-2/checkpoint_best.pt"
)
DEFAULT_CASE_DIR = Path(
    "/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_contribution_breakdown"
)


def _tensor_names_for_group(model) -> OrderedDict[str, list[str]]:
    names: OrderedDict[str, list[str]] = OrderedDict()
    if getattr(model, "head", None) is not None:
        names["head"] = [name for name, _ in model.head.named_parameters()]
    for block_idx, block in enumerate(model.blocks):
        names[f"block_{block_idx}/ff"] = [name for name, _ in block.ff.named_parameters()]
        names[f"block_{block_idx}/drive"] = ["drive_scale"]
        drn_names: list[str] = []
        weight_index = 0
        bias_index = 0
        for param in block.resistive_params():
            if isinstance(param, Bias):
                drn_names.append(f"b{bias_index}")
                bias_index += 1
            else:
                drn_names.append(f"W{weight_index}")
                weight_index += 1
        names[f"block_{block_idx}/drn"] = drn_names
    return names


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    na = float(a.norm().item())
    if na < 1.0e-12:
        return float("nan")
    return float((a - b).norm().item() / na)


def _merge_suffix(
    bp_groups: OrderedDict[str, list[torch.Tensor]],
    ep_groups: OrderedDict[str, list[torch.Tensor]],
    suffix: str,
) -> dict:
    bp_tensors: list[torch.Tensor] = []
    ep_tensors: list[torch.Tensor] = []
    for group_name in bp_groups.keys():
        if group_name.endswith("/" + suffix):
            bp_tensors.extend(bp_groups[group_name])
            ep_tensors.extend(ep_groups[group_name])
    bp_flat = _flatten(bp_tensors)
    ep_flat = _flatten(ep_tensors)
    return {
        "cosine": _cosine(bp_flat, ep_flat),
        "relative_error": _relative_error(bp_flat, ep_flat),
        "bp_norm": float(bp_flat.norm().item()),
        "ep_norm": float(ep_flat.norm().item()),
    }


def _summarize_variant(
    cfg: dict,
    checkpoint_path: Path,
    *,
    beta: float,
    device: torch.device,
    amp_gradient_compensation: bool,
    num_iterations: int | None = None,
) -> dict:
    model = build_model_from_config(cfg)
    model.set_device(device)
    _load_checkpoint_into_model(model, checkpoint_path)
    model.enable_resistive_grad_()
    model.eval()
    model.detach_state_()

    _train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    criterion = build_criterion(cfg.get("trainer", {}).get("criterion", "cross_entropy")).to(device)

    bp_totals = None
    ep_totals = None
    total_samples = 0
    tensor_names = _tensor_names_for_group(model)

    for batch_inputs, batch_targets in eval_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_size = int(batch_targets.size(0))
        bp_groups, ep_groups, displacement, _hybrid = _analyze_batch(
            model,
            batch_inputs,
            batch_targets,
            criterion,
            beta,
            num_iterations,
            amp_gradient_compensation=amp_gradient_compensation,
        )
        if bp_totals is None:
            bp_totals = _group_template(bp_groups)
            ep_totals = _group_template(ep_groups)
            displacement_totals = {
                "positive_relative_disp": 0.0,
                "negative_relative_disp": 0.0,
                "mean_relative_disp": 0.0,
            }
        _accumulate_group_sums(bp_totals, bp_groups, weight=float(batch_size))
        _accumulate_group_sums(ep_totals, ep_groups, weight=float(batch_size))
        for key in displacement_totals.keys():
            displacement_totals[key] += displacement[key] * batch_size
        total_samples += batch_size

    bp_means = _finalize_group_means(bp_totals, float(total_samples))
    ep_means = _finalize_group_means(ep_totals, float(total_samples))
    group_rows = _group_metrics(bp_means, ep_means)
    overall = _overall_metrics(bp_means, ep_means)
    path_rows = {
        "head": {
            "cosine": next(row["cosine"] for row in group_rows if row["group"] == "head"),
            "relative_error": next(row["relative_error"] for row in group_rows if row["group"] == "head"),
        }
        if "head" in bp_means
        else None,
        "ff_all": _merge_suffix(bp_means, ep_means, "ff"),
        "drive_all": _merge_suffix(bp_means, ep_means, "drive"),
        "drn_all": _merge_suffix(bp_means, ep_means, "drn"),
    }

    tensor_rows: list[dict] = []
    for group_name, names in tensor_names.items():
        for tensor_name, bp_tensor, ep_tensor in zip(names, bp_means[group_name], ep_means[group_name]):
            tensor_rows.append(
                {
                    "group": group_name,
                    "name": tensor_name,
                    "cosine": _cosine(bp_tensor, ep_tensor),
                    "relative_error": _relative_error(bp_tensor, ep_tensor),
                    "bp_norm": float(bp_tensor.norm().item()),
                    "ep_norm": float(ep_tensor.norm().item()),
                }
            )

    mean_displacement = {key: value / float(total_samples) for key, value in displacement_totals.items()}
    return {
        "overall": overall,
        "displacement": mean_displacement,
        "path_rows": path_rows,
        "group_rows": group_rows,
        "tensor_rows": tensor_rows,
    }


def _variant_config(
    base_cfg: dict,
    *,
    voltage_amp: float,
    current_amp: float,
    amplify_first_free_layer: bool,
) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("model", {}).setdefault("config", {})
    cfg["model"]["config"]["voltage_amp"] = voltage_amp
    cfg["model"]["config"]["current_amp"] = current_amp
    cfg["model"]["config"]["amplify_first_free_layer"] = amplify_first_free_layer
    for block in cfg["model"]["blocks_config"]:
        if "drn" not in block:
            continue
        block["drn"]["voltage_amp"] = voltage_amp
        block["drn"]["current_amp"] = current_amp
        block["drn"]["amplify_first_free_layer"] = amplify_first_free_layer
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down BP-vs-EP contributions on the wider hard-sigmoid EP checkpoint.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(args.config_path.read_text())
    set_seed(int(base_cfg.get("config", {}).get("seed", 0)))
    device = resolve_device(args.device)

    variants = [
        {
            "name": "va1_comp_on_layer1_on",
            "beta_values": [1.0e-3, 1.0e-2],
            "amp_gradient_compensation": True,
            "cfg": _variant_config(base_cfg, voltage_amp=1.0, current_amp=1.0, amplify_first_free_layer=True),
        },
        {
            "name": "va4_comp_on_layer1_on",
            "beta_values": [1.0e-3, 1.0e-2],
            "amp_gradient_compensation": True,
            "cfg": _variant_config(base_cfg, voltage_amp=4.0, current_amp=1.0, amplify_first_free_layer=True),
        },
        {
            "name": "va4_comp_off_layer1_on",
            "beta_values": [1.0e-3, 1.0e-2],
            "amp_gradient_compensation": False,
            "cfg": _variant_config(base_cfg, voltage_amp=4.0, current_amp=1.0, amplify_first_free_layer=True),
        },
        {
            "name": "va4_comp_on_layer1_off",
            "beta_values": [1.0e-3, 1.0e-2],
            "amp_gradient_compensation": True,
            "cfg": _variant_config(base_cfg, voltage_amp=4.0, current_amp=1.0, amplify_first_free_layer=False),
        },
        {
            "name": "va4_comp_off_layer1_off",
            "beta_values": [1.0e-3, 1.0e-2],
            "amp_gradient_compensation": False,
            "cfg": _variant_config(base_cfg, voltage_amp=4.0, current_amp=1.0, amplify_first_free_layer=False),
        },
    ]

    summary = {
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(device),
        "variants": [],
    }

    for variant in variants:
        variant_result = {
            "name": variant["name"],
            "voltage_amp": float(variant["cfg"]["model"]["config"]["voltage_amp"]),
            "current_amp": float(variant["cfg"]["model"]["config"]["current_amp"]),
            "amplify_first_free_layer": bool(variant["cfg"]["model"]["config"]["amplify_first_free_layer"]),
            "amp_gradient_compensation": bool(variant["amp_gradient_compensation"]),
            "rows": [],
        }
        print(
            f"[variant] {variant['name']} va={variant_result['voltage_amp']} "
            f"amp_comp={variant_result['amp_gradient_compensation']} "
            f"layer1_amp={variant_result['amplify_first_free_layer']}"
        )
        for beta in variant["beta_values"]:
            print(f"  [beta] {beta:g}")
            result = _summarize_variant(
                variant["cfg"],
                args.checkpoint_path,
                beta=beta,
                device=device,
                amp_gradient_compensation=variant["amp_gradient_compensation"],
            )
            variant_result["rows"].append({"beta": beta, **result})
            print(
                "    "
                f"overall_cos={result['overall']['overall_cosine']:.4f} "
                f"ff_cos={result['path_rows']['ff_all']['cosine']:.4f} "
                f"drn_cos={result['path_rows']['drn_all']['cosine']:.4f} "
                f"drn_relerr={result['path_rows']['drn_all']['relative_error']:.2f}"
            )
        summary["variants"].append(variant_result)

    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {case_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
