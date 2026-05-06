from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trial:
    name: str
    args: tuple[str, ...]


@dataclass
class TrialResult:
    name: str
    status: str
    cosine: float | None
    rel_mse: float | None
    loss: float | None
    output_dir: str
    metrics_path: str | None
    log_path: str
    command: str


TRIALS: tuple[Trial, ...] = (
    Trial("signed_lr1e4_clip1", ("--lr", "1e-4", "--grad_clip", "1.0")),
    Trial("signed_lr3e4_clip1", ("--lr", "3e-4", "--grad_clip", "1.0")),
    Trial("signed_lr1e3_clip1", ("--lr", "1e-3", "--grad_clip", "1.0")),
    Trial("signed_lr3e4_iter8", ("--lr", "3e-4", "--grad_clip", "1.0", "--drn_iter", "8")),
    Trial("signed_lr3e4_std_scale", ("--lr", "3e-4", "--grad_clip", "1.0", "--scale_source", "std")),
    Trial("signed_lr3e4_gain03", ("--lr", "3e-4", "--grad_clip", "1.0", "--drn_weight_gains", "0.3")),
    Trial("signed_teacher_frontend_lr3e4", ("--init_mode", "teacher_frontend_scale", "--lr", "3e-4", "--grad_clip", "1.0")),
    Trial(
        "signed_teacher_frontend_lr1e3_amp",
        ("--init_mode", "teacher_frontend_scale", "--lr", "1e-3", "--grad_clip", "1.0", "--drn_learn_amplification"),
    ),
    Trial("unsigned_lr3e4", ("--no-drn_signed_drive", "--lr", "3e-4", "--grad_clip", "1.0")),
    Trial(
        "teacher_frontend_lr3e4",
        ("--no-drn_signed_drive", "--init_mode", "teacher_frontend_scale", "--lr", "3e-4", "--grad_clip", "1.0"),
    ),
    Trial(
        "teacher_frontend_lr1e3_amp",
        (
            "--no-drn_signed_drive",
            "--init_mode",
            "teacher_frontend_scale",
            "--lr",
            "1e-3",
            "--grad_clip",
            "1.0",
            "--drn_learn_amplification",
        ),
    ),
    Trial(
        "hard_sigmoid_lr3e4",
        (
            "--drn_non_linearity",
            "hard_sigmoid",
            "--lr",
            "3e-4",
            "--grad_clip",
            "1.0",
            "--drn_learn_amplification",
        ),
    ),
)


def main() -> None:
    args = _parse_args()
    run_root = Path(args.output_dir) / f"opt_mlp_drn_hyperopt_layer{args.layer}_{_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    selected_trials = TRIALS[: args.max_trials] if args.max_trials is not None else TRIALS
    summary_path = run_root / "summary.jsonl"
    rows: list[TrialResult] = []

    for trial in selected_trials:
        result = _run_trial(args, trial, run_root)
        rows.append(result)
        _append_jsonl(summary_path, asdict(result))
        _write_csv(run_root / "summary.csv", rows)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
        if result.cosine is not None and result.cosine >= args.target_cosine:
            break

    best = max((row for row in rows if row.cosine is not None), key=lambda row: row.cosine, default=None)
    final = {"best": asdict(best) if best is not None else None, "target_cosine": args.target_cosine}
    (run_root / "final_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


def _run_trial(args: argparse.Namespace, trial: Trial, run_root: Path) -> TrialResult:
    trial_root = run_root / trial.name
    trial_root.mkdir(parents=True, exist_ok=True)
    log_path = trial_root / "trial.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "opt_mlp_drn.single_block_train",
        "--model_name",
        args.model_name,
        "--data",
        str(args.data),
        "--layers",
        str(args.layer),
        "--objective",
        args.objective,
        "--mlp_input_mode",
        args.mlp_input_mode,
        "--block_size",
        str(args.block_size),
        "--batch_size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_iters",
        str(args.eval_iters),
        "--calibration_batches",
        str(args.calibration_batches),
        "--eval_logit_kl_batches",
        str(args.eval_logit_kl_batches),
        "--drn_iter",
        str(args.drn_iter),
        "--drn_drive_architecture",
        args.drn_drive_architecture,
        "--weight_decay",
        str(args.weight_decay),
        "--output_dir",
        str(trial_root),
        "--device",
        args.device,
        *trial.args,
    ]
    if args.activation_cache is not None:
        command.extend(["--activation_cache", str(args.activation_cache)])
    if args.drn_amp_lr is not None:
        command.extend(["--drn_amp_lr", str(args.drn_amp_lr)])
    if args.tokenizer != "auto":
        command.extend(["--tokenizer", args.tokenizer])
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=args.cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    metrics_path = _find_metrics_path(trial_root, args.layer)
    metrics = _read_metrics(metrics_path) if metrics_path is not None else {}
    return TrialResult(
        name=trial.name,
        status="ok" if completed.returncode == 0 and metrics_path is not None else f"failed:{completed.returncode}",
        cosine=_as_float(metrics.get("cosine")),
        rel_mse=_as_float(metrics.get("rel_mse")),
        loss=_as_float(metrics.get("loss")),
        output_dir=str(trial_root),
        metrics_path=str(metrics_path) if metrics_path is not None else None,
        log_path=str(log_path),
        command=" ".join(command),
    )


def _find_metrics_path(trial_root: Path, layer: int) -> Path | None:
    candidates = sorted(trial_root.glob(f"opt_mlp_drn_single_block_*/layer_{layer}/final_metrics.json"))
    return candidates[-1] if candidates else None


def _read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[TrialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=Path("data/tiny_shakespeare.txt"))
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--activation_cache", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--target_cosine", type=float, default=0.95)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument(
        "--objective",
        choices=["local_mlp", "local_mlp_cosine", "post_residual", "next_ln_aux"],
        default="local_mlp",
    )
    parser.add_argument("--mlp_input_mode", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=5)
    parser.add_argument("--calibration_batches", type=int, default=8)
    parser.add_argument("--eval_logit_kl_batches", type=int, default=0)
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--drn_amp_lr", type=float, default=None)
    parser.add_argument(
        "--drn_drive_architecture",
        choices=["projected_hidden", "signed_input_free"],
        default="projected_hidden",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", type=Path, default=Path("simulation_results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    return parser.parse_args()


if __name__ == "__main__":
    main()
