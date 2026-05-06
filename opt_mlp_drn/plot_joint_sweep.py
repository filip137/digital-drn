from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    args = _parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    teacher_ppl = None
    summary = {}
    for label, metrics_path in [_parse_run(raw) for raw in args.run]:
        rows = _eval_rows(metrics_path)
        steps = [int(row["step"]) for row in rows if "student_ppl" in row]
        ppls = [float(row["student_ppl"]) for row in rows if "student_ppl" in row]
        if not steps:
            raise ValueError(f"No eval rows with student_ppl in {metrics_path}.")
        ax.plot(steps, ppls, marker="o", linewidth=2.0, label=label)
        final = rows[-1]
        summary[label] = {
            "final_step": int(final["step"]),
            "student_ppl": float(final["student_ppl"]),
            "student_ce_loss": float(final["student_ce_loss"]),
            "logit_kl": float(final["logit_kl"]),
            "hidden_final_rel_rms": float(final["hidden_final_rel_rms"]),
        }
        if teacher_ppl is None and "teacher_ppl" in final:
            teacher_ppl = float(final["teacher_ppl"])

    if teacher_ppl is not None:
        ax.axhline(teacher_ppl, color="black", linestyle="--", linewidth=1.5, label=f"digital baseline ({teacher_ppl:.3f})")
    ax.set_xlabel("joint training step")
    ax.set_ylabel("validation perplexity")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    if args.summary is not None:
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _eval_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == "eval":
            rows.append(row)
    return rows


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--run entries must have the form LABEL=METRICS_JSONL.")
    label, path = raw.split("=", 1)
    return label, Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--title", default="OPT last-3 MLP DRN recovery")
    return parser.parse_args()


if __name__ == "__main__":
    main()
