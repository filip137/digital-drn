from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InitVariant:
    name: str
    structural_init: str
    post_drn_alpha_init: float
    post_ladder_lambda_init: float
    note: str


INIT_VARIANTS: tuple[InitVariant, ...] = (
    InitVariant(
        name="random_alpha1_lambda1",
        structural_init="none",
        post_drn_alpha_init=1.0,
        post_ladder_lambda_init=1.0,
        note="Random side/DRN init with full post-DRN ladder scale.",
    ),
    InitVariant(
        name="selector_alpha1_lambda1",
        structural_init="magnitude",
        post_drn_alpha_init=1.0,
        post_ladder_lambda_init=1.0,
        note="Magnitude-selected projection init inspired by upstream pruned side weights.",
    ),
    InitVariant(
        name="selector_attn_alpha1_lambda1",
        structural_init="magnitude_attn",
        post_drn_alpha_init=1.0,
        post_ladder_lambda_init=1.0,
        note="Magnitude-selected projections plus pruned GPT-2 attention/LN weights in the side attention path.",
    ),
    InitVariant(
        name="selector_attn_alpha01_lambda1",
        structural_init="magnitude_attn",
        post_drn_alpha_init=0.1,
        post_ladder_lambda_init=1.0,
        note="Pruned attention init with a conservative DRN residual scale.",
    ),
    InitVariant(
        name="selector_attn_alpha05_lambda05",
        structural_init="magnitude_attn",
        post_drn_alpha_init=0.5,
        post_ladder_lambda_init=0.5,
        note="Pruned attention init with balanced DRN and ladder scales.",
    ),
    InitVariant(
        name="selector_attn_alpha1_lambda05",
        structural_init="magnitude_attn",
        post_drn_alpha_init=1.0,
        post_ladder_lambda_init=0.5,
        note="Pruned attention init with a smaller post-DRN ladder injection.",
    ),
)


def main() -> None:
    args = _parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "sweep_config.json", _serializable_vars(args))

    if args.stage in {"all", "init"}:
        run_init_sweep(args, root)

    if args.stage in {"all", "optuna"}:
        run_optuna(args, root)


def run_init_sweep(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    init_root = root / "init_sweep"
    init_root.mkdir(parents=True, exist_ok=True)
    for variant in INIT_VARIANTS:
        run_parent = init_root / variant.name
        selected_run_path = run_parent / "selected_run.json"
        if selected_run_path.exists() and not args.force:
            row = _read_json(selected_run_path)
            rows.append(row)
            print(f"[init] reusing {variant.name}: best_val_loss={row.get('best_val_loss')}", flush=True)
            continue

        extra_args = _variant_args(variant)
        print(f"[init] running {variant.name}: {variant.note}", flush=True)
        row = _run_train(args, run_parent, extra_args, run_label=f"init:{variant.name}")
        row.update({"variant": asdict(variant), "variant_name": variant.name})
        _write_json(selected_run_path, row)
        rows.append(row)

    rows = sorted(rows, key=lambda item: _metric_or_inf(item, "best_val_loss"))
    summary = {"rows": rows, "best": rows[0] if rows else None}
    _write_json(root / "init_sweep_summary.json", summary)
    if rows:
        _write_json(root / "best_init.json", rows[0])
        print(
            f"[init] best={rows[0]['variant_name']} "
            f"best_val_loss={rows[0].get('best_val_loss')}",
            flush=True,
        )
    return summary


def run_optuna(args: argparse.Namespace, root: Path) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for the LR search. Install optuna in the active environment.") from exc

    best_init_path = root / "best_init.json"
    if not best_init_path.exists():
        raise RuntimeError(f"Missing {best_init_path}; run the init stage first or provide an existing output root.")
    best_init = _read_json(best_init_path)
    variant = InitVariant(**best_init["variant"])

    optuna_root = root / "optuna_lr"
    optuna_root.mkdir(parents=True, exist_ok=True)
    storage_path = optuna_root / "study.db"
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        lr = trial.suggest_float("lr", args.lr_low, args.lr_high, log=True)
        lr_decay = trial.suggest_categorical("lr_decay", ["none", "cosine", "linear"])
        if lr_decay == "none":
            warmup_steps = 0
            min_lr_ratio = 0.0
        else:
            warmup_steps = trial.suggest_categorical("warmup_steps", [0, 50, 100, 250, 500])
            min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.0, 0.5)
        min_lr = lr * min_lr_ratio

        trial_root = optuna_root / f"trial_{trial.number:04d}"
        extra_args = _variant_args(variant) + [
            "--lr",
            _fmt_float(lr),
            "--lr_decay",
            lr_decay,
            "--warmup_steps",
            str(warmup_steps),
            "--min_lr",
            _fmt_float(min_lr),
        ]
        _write_json(
            trial_root / "trial_config.json",
            {
                "trial": trial.number,
                "best_init": best_init,
                "lr": lr,
                "lr_decay": lr_decay,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "min_lr": min_lr,
            },
        )
        row = _run_train(args, trial_root, extra_args, run_label=f"optuna:{trial.number}")
        metric = _metric_or_inf(row, "best_val_loss")
        trial.set_user_attr("run_dir", row.get("run_dir"))
        trial.set_user_attr("peak_memory_mb", row.get("peak_memory_mb"))
        trial.set_user_attr("trainable_params", row.get("trainable_params"))
        trial.set_user_attr("total_params", row.get("total_params"))
        trial.set_user_attr("best_init_variant", variant.name)
        trial.set_user_attr("result", row)
        _write_json(trial_root / "trial_result.json", row)
        print(f"[optuna] trial={trial.number} metric={metric:.6f}", flush=True)
        if not math.isfinite(metric):
            raise RuntimeError(f"Trial {trial.number} returned a non-finite metric.")
        return metric

    timeout = int(args.optuna_timeout_hours * 3600)
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=timeout,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        _write_json(optuna_root / "best_trial.json", {"best_init": best_init, "error": "no_completed_trials"})
        print("[optuna] no completed trials", flush=True)
        return

    best_trial = min(completed, key=lambda trial: float(trial.value))
    best = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": dict(best_trial.user_attrs),
        "best_init": best_init,
    }
    _write_json(optuna_root / "best_trial.json", best)
    print(f"[optuna] best_trial={best['number']} value={best['value']} params={best['params']}", flush=True)


def _variant_args(variant: InitVariant) -> list[str]:
    return [
        "--lst_structural_init",
        variant.structural_init,
        "--post_drn_alpha_init",
        _fmt_float(variant.post_drn_alpha_init),
        "--post_ladder_lambda_init",
        _fmt_float(variant.post_ladder_lambda_init),
    ]


def _run_train(
    args: argparse.Namespace,
    run_parent: Path,
    extra_args: list[str],
    *,
    run_label: str,
) -> dict[str, Any]:
    run_parent.mkdir(parents=True, exist_ok=True)
    log_path = run_parent / "train.log"
    common = _common_train_args(args, run_parent)
    cmd = [args.python, "-m", "gpt2_ladder_drn.train", *common, *extra_args]
    _write_json(run_parent / "command.json", {"cmd": cmd, "cwd": str(args.repo_root), "label": run_label})
    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return {"best_val_loss": float("inf"), "run_dir": None, "dry_run": True}

    print(f"[run] {run_label} -> {run_parent}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=args.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_child_env(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training command failed with exit code {return_code}; see {log_path}")

    run_dir = _latest_run_dir(run_parent)
    final_metrics = _read_json(run_dir / "final_metrics.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    row = {
        "run_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
        "best_val_loss": final_metrics.get("best_val_loss"),
        "peak_memory_mb": final_metrics.get("peak_memory_mb"),
        "trainable_params": metadata.get("trainable_params"),
        "total_params": metadata.get("total_params"),
        "metadata": metadata,
    }
    _write_json(run_parent / "result.json", row)
    return row


def _common_train_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        "--mode",
        "lst_drn",
        "--pretrained",
        args.pretrained,
        "--tokenizer",
        args.tokenizer,
        "--data",
        str(args.data),
        "--block_size",
        str(args.block_size),
        "--batch_size",
        str(args.batch_size),
        "--eval_iters",
        str(args.eval_iters),
        "--eval_interval",
        str(args.eval_interval),
        "--max_steps",
        str(args.max_steps),
        "--weight_decay",
        _fmt_float(args.weight_decay),
        "--grad_clip",
        _fmt_float(args.grad_clip),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--sample_tokens",
        "0",
        "--lst_reduction",
        "4",
        "--lst_side_layers",
        "4",
        "--lst_tap_indices",
        args.lst_tap_indices,
        "--lst_temperature",
        _fmt_float(args.lst_temperature),
        "--lst_output_mode",
        "gated_logits",
        "--logit_gate_alpha_init",
        _fmt_float(args.logit_gate_alpha_init),
        "--lst_initial_state_mode",
        "full_tap",
        "--lst_side_block_type",
        "drn_hybrid_attn",
        "--ladder_injection_mode",
        "post_drn_residual",
        "--backbone_tap_kind",
        "block_output",
        "--post_drn_feedback",
        "--drn_signed_drive",
        "--drn_hidden_multiplier",
        str(args.drn_hidden_multiplier),
        "--drn_iter",
        str(args.drn_iter),
        "--drn_damping",
        _fmt_float(args.drn_damping),
        "--lr",
        _fmt_float(args.lr),
        "--lr_decay",
        args.lr_decay,
        "--warmup_steps",
        str(args.warmup_steps),
        "--min_lr",
        _fmt_float(args.min_lr),
    ]


def _latest_run_dir(run_parent: Path) -> Path:
    candidates = sorted(
        (path for path in run_parent.glob("lst_drn_*") if (path / "final_metrics.json").exists()),
        key=lambda path: (path / "final_metrics.json").stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No completed train run found under {run_parent}")
    return candidates[-1]


def _metric_or_inf(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return metric if math.isfinite(metric) else float("inf")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serializable_vars(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _fmt_float(value: float) -> str:
    return f"{float(value):.12g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the r=4/depth=4 GPT-2 DRN ladder post-DRN-residual init sweep, "
            "then tune learning-rate settings with Optuna."
        )
    )
    parser.add_argument("--stage", choices=["all", "init", "optuna"], default="all")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--pretrained", default="gpt2")
    parser.add_argument("--tokenizer", choices=["gpt2", "char"], default="gpt2")
    parser.add_argument("--data", type=Path, default=Path("data/tiny_shakespeare.txt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--max_steps", type=int, default=2000)

    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--lr_decay", choices=["none", "cosine", "linear"], default="none")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lst_tap_indices", default="3,6,9,12")
    parser.add_argument("--lst_temperature", type=float, default=0.1)
    parser.add_argument("--logit_gate_alpha_init", type=float, default=0.0)
    parser.add_argument("--drn_hidden_multiplier", type=int, default=8)
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--drn_damping", type=float, default=0.5)

    parser.add_argument("--study_name", default="gpt2_post_drn_r4_depth4_lr")
    parser.add_argument("--optuna_trials", type=int, default=200)
    parser.add_argument("--optuna_timeout_hours", type=float, default=12.0)
    parser.add_argument("--lr_low", type=float, default=5.0e-5)
    parser.add_argument("--lr_high", type=float, default=3.0e-3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
