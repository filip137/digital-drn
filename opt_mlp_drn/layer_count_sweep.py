from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("simulation_results/opt_mlp_drn_layer_count_kl_20260505")
DEFAULT_CASE_DIR = Path("cases/transformer_cases/opt_mlp_drn_recovery_20260505")

DEFAULT_TRAIN_DATA = Path("simulation_results/opt_mlp_drn_synthetic_splits_20260505/train_100k.txt")
DEFAULT_VAL_DATA = Path("simulation_results/opt_mlp_drn_synthetic_splits_20260505/val_20k.txt")
DEFAULT_TEST_DATA = Path("simulation_results/opt_mlp_drn_synthetic_splits_20260505/test_20k.txt")

CHECKPOINTS = {
    9: Path(
        "simulation_results/opt_mlp_drn_phase5_layer9_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/"
        "opt_mlp_drn_single_block_20260505-173625/layer_9/checkpoint_last.pt"
    ),
    10: Path(
        "simulation_results/opt_mlp_drn_phase5_layer10_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/"
        "opt_mlp_drn_single_block_20260505-173344/layer_10/checkpoint_last.pt"
    ),
    11: Path(
        "simulation_results/opt_mlp_drn_phase5_layer11_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/"
        "opt_mlp_drn_single_block_20260505-173054/layer_11/checkpoint_last.pt"
    ),
}


@dataclass(frozen=True)
class SweepJob:
    job_id: str
    layer_count: int
    replaced_layers: tuple[int, ...]
    trainable_policy: str
    replace_mlp_layers: str
    trainable_scope: str
    train_final_ln: bool


@dataclass
class SweepResult:
    job_id: str
    layer_count: int
    replaced_layers: list[int]
    trainable_policy: str
    trainable_scope: str
    run_dir: str | None
    initial_val_kl: float | None
    best_val_kl: float | None
    test_kl: float | None
    student_test_ppl: float | None
    teacher_test_ppl: float | None
    final_hidden_rel_rms: float | None
    test_hidden_rel_rms: float | None
    trainable_params: int | None
    trainable_embedding_params: int | None
    trainable_lm_head_params: int | None
    peak_memory_mb: float | None
    checkpoint_paths_init: list[str]
    complete: bool


def build_jobs() -> list[SweepJob]:
    jobs: list[SweepJob] = []
    layer_sets = [
        ("last1", 1, (11,), "last:1"),
        ("last2", 2, (10, 11), "last:2"),
        ("last3", 3, (9, 10, 11), "last:3"),
    ]
    for suffix, count, layers, selector in layer_sets:
        jobs.append(
            SweepJob(
                job_id=f"local_{suffix}",
                layer_count=count,
                replaced_layers=layers,
                trainable_policy="local_recovery",
                replace_mlp_layers=selector,
                trainable_scope="drn_attn_full_ln",
                train_final_ln=True,
            )
        )
    for suffix, count, layers, selector in layer_sets:
        jobs.append(
            SweepJob(
                job_id=f"whole_{suffix}",
                layer_count=count,
                replaced_layers=layers,
                trainable_policy="whole_network",
                replace_mlp_layers=selector,
                trainable_scope="all_student",
                train_final_ln=False,
            )
        )
    return jobs


def main() -> None:
    args = _parse_args()
    jobs = _selected_jobs(args)
    if args.command in {"run", "all"}:
        for job in jobs:
            _run_job(job, args)
    if args.command in {"summarize", "all"}:
        rows = summarize(args.output_root, jobs)
        write_summary(args.output_root, rows)
        write_plots(args.output_root, rows)
        write_case_markdown(args.case_dir, args.output_root, rows)
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))


def _run_job(job: SweepJob, args: argparse.Namespace) -> None:
    job_root = args.output_root / job.job_id
    if _latest_run_dir(job_root) is not None and not args.force:
        print(f"[skip] {job.job_id}: existing run found under {job_root}", flush=True)
        return
    command = build_command(job, args, job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    print(f"[run] {job.job_id}", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def build_command(job: SweepJob, args: argparse.Namespace, job_root: Path) -> list[str]:
    command = [
        args.python,
        "-u",
        "-m",
        "opt_mlp_drn.joint_train",
        "--model_name",
        args.model_name,
        "--train_data",
        str(args.train_data),
        "--val_data",
        str(args.val_data),
        "--test_data",
        str(args.test_data),
        "--tokenizer",
        "auto",
        "--replace_mlp_layers",
        job.replace_mlp_layers,
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
        "--early_stopping_metric",
        "logit_kl",
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--distill_objective",
        "logit_kl",
        "--distill_beta",
        "1.0",
        "--kl_temperature",
        "1.0",
        "--trainable_scope",
        job.trainable_scope,
        "--lr",
        str(args.lr),
        "--attn_lr",
        str(args.attn_lr),
        "--ln_lr",
        str(args.ln_lr),
        "--weight_decay",
        "0.0",
        "--grad_clip",
        str(args.grad_clip),
        "--drn_iter",
        str(args.drn_iter),
        "--drn_drive_architecture",
        "signed_input_free",
        "--drn_learn_amplification",
        "--drn_amp_lr",
        str(args.drn_amp_lr),
        "--output_dir",
        str(job_root),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if job.train_final_ln:
        command.append("--train_final_ln")
    for layer in job.replaced_layers:
        checkpoint = CHECKPOINTS[layer]
        command.extend(["--checkpoint_path", f"{layer}={checkpoint}"])
    return command


def summarize(output_root: Path, jobs: list[SweepJob]) -> list[SweepResult]:
    rows = []
    for job in jobs:
        rows.append(_summarize_job(output_root, job))
    return rows


def _summarize_job(output_root: Path, job: SweepJob) -> SweepResult:
    run_dir = _latest_run_dir(output_root / job.job_id)
    if run_dir is None:
        return SweepResult(
            job_id=job.job_id,
            layer_count=job.layer_count,
            replaced_layers=list(job.replaced_layers),
            trainable_policy=job.trainable_policy,
            trainable_scope=job.trainable_scope,
            run_dir=None,
            initial_val_kl=None,
            best_val_kl=None,
            test_kl=None,
            student_test_ppl=None,
            teacher_test_ppl=None,
            final_hidden_rel_rms=None,
            test_hidden_rel_rms=None,
            trainable_params=None,
            trainable_embedding_params=None,
            trainable_lm_head_params=None,
            peak_memory_mb=None,
            checkpoint_paths_init=[f"{layer}={CHECKPOINTS[layer]}" for layer in job.replaced_layers],
            complete=False,
        )
    final_path = run_dir / "final_metrics.json"
    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "metrics.jsonl"
    complete = final_path.exists() and metadata_path.exists()
    final = _read_json(final_path) if final_path.exists() else {}
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    initial = _initial_eval(metrics_path)
    return SweepResult(
        job_id=job.job_id,
        layer_count=job.layer_count,
        replaced_layers=list(job.replaced_layers),
        trainable_policy=job.trainable_policy,
        trainable_scope=str(metadata.get("trainable_scope", job.trainable_scope)),
        run_dir=str(run_dir),
        initial_val_kl=_float_or_none(initial.get("logit_kl")),
        best_val_kl=_float_or_none(final.get("best_validation_metric")),
        test_kl=_float_or_none(final.get("test_logit_kl")),
        student_test_ppl=_float_or_none(final.get("test_student_ppl", final.get("test_ppl"))),
        teacher_test_ppl=_float_or_none(final.get("test_teacher_ppl")),
        final_hidden_rel_rms=_float_or_none(final.get("hidden_final_rel_rms")),
        test_hidden_rel_rms=_float_or_none(final.get("test_hidden_final_rel_rms")),
        trainable_params=_int_or_none(metadata.get("trainable_params")),
        trainable_embedding_params=_int_or_none(metadata.get("trainable_embedding_params")),
        trainable_lm_head_params=_int_or_none(metadata.get("trainable_lm_head_params")),
        peak_memory_mb=_float_or_none(final.get("peak_memory_mb")),
        checkpoint_paths_init=list(metadata.get("checkpoint_paths_init") or [f"{layer}={CHECKPOINTS[layer]}" for layer in job.replaced_layers]),
        complete=complete,
    )


def write_summary(output_root: Path, rows: list[SweepResult]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    (output_root / "layer_count_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_root / "layer_count_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)
    (output_root / "layer_count_summary.md").write_text(_markdown_table(rows), encoding="utf-8")


def write_plots(output_root: Path, rows: list[SweepResult]) -> None:
    complete = [row for row in rows if row.complete and row.test_kl is not None]
    if not complete:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _line_plot(
        complete,
        value_name="test_kl",
        ylabel="test KL teacher||student (nats/token)",
        title="OPT DRN layer-count sweep: test KL",
        output=output_root / "layer_count_test_kl.png",
    )
    _line_plot(
        [row for row in complete if row.student_test_ppl is not None],
        value_name="student_test_ppl",
        ylabel="student test perplexity",
        title="OPT DRN layer-count sweep: test PPL",
        output=output_root / "layer_count_test_ppl.png",
        teacher_ppl=next((row.teacher_test_ppl for row in complete if row.teacher_test_ppl is not None), None),
    )
    ratio_rows = [row for row in complete if row.student_test_ppl is not None and row.teacher_test_ppl]
    if ratio_rows:
        _line_plot(
            ratio_rows,
            value_name="ppl_ratio",
            ylabel="student PPL / teacher PPL",
            title="OPT DRN layer-count sweep: PPL ratio",
            output=output_root / "layer_count_ppl_ratio.png",
        )


def write_case_markdown(case_dir: Path, output_root: Path, rows: list[SweepResult]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    content = [
        "# OPT DRN Layer-Count KL Sweep",
        "",
        "This file is generated by `python -m opt_mlp_drn.layer_count_sweep summarize`.",
        "",
        f"Output root: `{output_root}`",
        "",
        "## Results",
        "",
        _markdown_table(rows),
        "",
        "## Plots",
        "",
        f"- `{output_root / 'layer_count_test_kl.png'}`",
        f"- `{output_root / 'layer_count_test_ppl.png'}`",
        f"- `{output_root / 'layer_count_ppl_ratio.png'}`",
        "",
    ]
    (case_dir / "layer_count_kl_sweep.md").write_text("\n".join(content), encoding="utf-8")


def _line_plot(rows: list[SweepResult], *, value_name: str, ylabel: str, title: str, output: Path, teacher_ppl: float | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for policy, label in [("local_recovery", "local recovery"), ("whole_network", "whole network")]:
        selected = sorted([row for row in rows if row.trainable_policy == policy], key=lambda row: row.layer_count)
        if not selected:
            continue
        xs = [row.layer_count for row in selected]
        if value_name == "ppl_ratio":
            ys = [float(row.student_test_ppl) / float(row.teacher_test_ppl) for row in selected]
        else:
            ys = [float(getattr(row, value_name)) for row in selected]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    if teacher_ppl is not None:
        ax.axhline(float(teacher_ppl), color="black", linestyle="--", linewidth=1.2, label=f"teacher ({teacher_ppl:.3f})")
    ax.set_xticks(sorted({row.layer_count for row in rows}))
    ax.set_xlabel("number of replaced DRN MLP layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[SweepResult]) -> str:
    headers = [
        "job",
        "layers",
        "policy",
        "init val KL",
        "best val KL",
        "test KL",
        "test PPL",
        "teacher PPL",
        "hidden RMS",
        "trainable params",
        "emb trainable",
        "LM trainable",
        "peak MB",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item.trainable_policy, item.layer_count)):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.job_id,
                    str(row.layer_count),
                    row.trainable_policy,
                    _fmt(row.initial_val_kl),
                    _fmt(row.best_val_kl),
                    _fmt(row.test_kl),
                    _fmt(row.student_test_ppl),
                    _fmt(row.teacher_test_ppl),
                    _fmt(row.test_hidden_rel_rms),
                    _fmt_int(row.trainable_params),
                    _fmt_int(row.trainable_embedding_params),
                    _fmt_int(row.trainable_lm_head_params),
                    _fmt(row.peak_memory_mb, digits=1),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _latest_run_dir(job_root: Path) -> Path | None:
    if not job_root.exists():
        return None
    candidates = sorted(path for path in job_root.glob("opt_mlp_drn_joint_*") if path.is_dir())
    return candidates[-1] if candidates else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_eval(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == "eval" and int(row.get("step", -1)) == 0:
            return row
    return {}


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _fmt(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{int(value):,}"


def _selected_jobs(args: argparse.Namespace) -> list[SweepJob]:
    jobs = build_jobs()
    if not args.jobs:
        return jobs
    wanted = set(args.jobs)
    unknown = sorted(wanted - {job.job_id for job in jobs})
    if unknown:
        raise ValueError(f"Unknown jobs: {unknown}. Available: {[job.job_id for job in jobs]}")
    return [job for job in jobs if job.job_id in wanted]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summarize", "all"])
    parser.add_argument("--jobs", nargs="*", default=[])
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case_dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val_data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--test_data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--attn_lr", type=float, default=1.0e-5)
    parser.add_argument("--ln_lr", type=float, default=1.0e-5)
    parser.add_argument("--drn_amp_lr", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
