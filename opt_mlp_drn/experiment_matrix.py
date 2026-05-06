from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    phase: str
    dataset_source: str
    token_budget: str
    loss: str
    init: str
    solver_iterations: int
    drn_hidden_dim: int
    trainable_mode: str
    command: list[str]


DATASET_SOURCES = ("real", "synthetic", "mixed")
TOKEN_BUDGETS = ("1M", "10M", "50M")
LOSS_PRESETS = {
    "local_mlp": ("opt_mlp_drn.single_block_train", ["--objective", "local_mlp", "--layers", "all"]),
    "post_residual": ("opt_mlp_drn.single_block_train", ["--objective", "post_residual", "--layers", "all"]),
    "post_residual_next_ln": ("opt_mlp_drn.single_block_train", ["--objective", "next_ln_aux", "--layers", "all"]),
    "progressive_residual": ("opt_mlp_drn.progressive_train", ["--layers", "all"]),
    "hidden_kl": ("opt_mlp_drn.joint_train", ["--replace_mlp_layers", "all", "--hidden_weight", "1.0", "--kl_weight", "0.1"]),
    "hidden_kl_ce": ("opt_mlp_drn.joint_train", ["--replace_mlp_layers", "all", "--hidden_weight", "1.0", "--kl_weight", "0.1", "--ce_weight", "0.1"]),
}
INITS = ("random", "teacher_frontend_scale", "teacher_frontend_no_scale", "teacher_frontend_trainable_scaling")
SOLVER_ITERS = (2, 4, 8)
CAPACITIES = (3072, 2048, 1536, 1024)
TRAINABLE_MODES = ("conductances", "output_readout", "scaling_only", "conductances_scaling", "drn_layernorm")


def build_matrix() -> list[ExperimentSpec]:
    rows: list[ExperimentSpec] = []
    for source in DATASET_SOURCES:
        rows.append(_spec("dataset", "phase8_dataset", source, "1M", "hidden_kl", "teacher_frontend_scale", 4, 3072, "conductances_scaling"))
    for budget in TOKEN_BUDGETS:
        rows.append(_spec("budget", "phase8_budget", "mixed", budget, "hidden_kl", "teacher_frontend_scale", 4, 3072, "conductances_scaling"))
    for loss in LOSS_PRESETS:
        rows.append(_spec("loss", "phase8_loss", "mixed", "1M", loss, "teacher_frontend_scale", 4, 3072, "conductances_scaling"))
    for init in INITS:
        rows.append(_spec("init", "phase8_init", "mixed", "1M", "local_mlp", init, 4, 3072, "conductances_scaling"))
    for iterations in SOLVER_ITERS:
        rows.append(_spec("solver", "phase8_solver", "mixed", "1M", "hidden_kl", "teacher_frontend_scale", iterations, 3072, "conductances_scaling"))
    for capacity in CAPACITIES:
        rows.append(_spec("capacity", "phase8_capacity", "mixed", "1M", "hidden_kl", "teacher_frontend_scale", 4, capacity, "conductances_scaling"))
    for trainable in TRAINABLE_MODES:
        rows.append(_spec("trainable", "phase8_trainable", "mixed", "1M", "hidden_kl", "teacher_frontend_scale", 4, 3072, trainable))
    return rows


def _spec(prefix: str, phase: str, source: str, budget: str, loss: str, init: str, iterations: int, capacity: int, trainable: str) -> ExperimentSpec:
    module, loss_args = LOSS_PRESETS[loss]
    name = f"{prefix}_{source}_{budget}_{loss}_{init}_iter{iterations}_h{capacity}_{trainable}"
    command = [
        "python",
        "-m",
        module,
        "--model_name",
        "facebook/opt-125m",
        "--data",
        f"data/opt_{source}_{budget.lower()}.txt",
        "--drn_iter",
        str(iterations),
        "--drn_hidden_multiplier",
        str(capacity / 768.0),
        "--output_dir",
        f"simulation_results/{name}",
        *loss_args,
    ]
    if module.endswith("single_block_train") and init != "teacher_frontend_trainable_scaling":
        command.extend(["--init_mode", init])
    if init == "teacher_frontend_trainable_scaling":
        command.extend(["--init_mode", "teacher_frontend_scale", "--drn_learn_amplification"])
    if trainable in {"scaling_only", "conductances_scaling"}:
        command.append("--drn_learn_amplification")
    return ExperimentSpec(name, phase, source, budget, loss, init, iterations, capacity, trainable, command)


def main() -> None:
    args = _parse_args()
    rows = build_matrix()
    if args.list:
        for row in rows:
            print(row.name)
    if args.write_plan is not None:
        _write_plan(args.write_plan, rows)


def _write_plan(path: Path, rows: list[ExperimentSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                payload = asdict(row)
                payload["command"] = " ".join(row.command)
                writer.writerow(payload)
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--write_plan", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
