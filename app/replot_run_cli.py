from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


PERFORMANCE_FILES = {
    "train/loss": Path("performance") / "train_loss.png",
    "train/accuracy": Path("performance") / "train_accuracy.png",
    "train/learning_rate": Path("performance") / "train_learning_rate.png",
    "val/loss": Path("performance") / "val_loss.png",
    "val/accuracy": Path("performance") / "val_accuracy.png",
}

COMBINED_ACCURACY_FILE = Path("performance") / "train_test_accuracy.png"
TITLE_FONTSIZE = 28
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def _existing_plot_relpaths(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    return {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*.png")
        if path.is_file()
    }


def _find_event_file(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}.")
    return candidates[0]


def _tag_to_relpath(tag: str) -> Path:
    if tag in PERFORMANCE_FILES:
        return PERFORMANCE_FILES[tag]
    group = tag.split("/", 1)[0]
    return Path(group) / f"{tag.replace('/', '_')}.png"


def _pretty_metric_name(tag: str) -> str:
    parts = tag.split("/")
    if len(parts) == 2 and parts[0] in {"train", "val"}:
        phase = parts[0].upper()
        metric = parts[1].replace("_", " ").title()
        return f"{phase} {metric}"
    return tag.replace("/", " / ").replace("_", " ")


def _ylabel(tag: str) -> str:
    if tag.endswith("/accuracy") or tag == "train/accuracy" or tag == "val/accuracy":
        return "Accuracy"
    if tag.endswith("/loss") or tag == "train/loss" or tag == "val/loss":
        return "Loss"
    if tag.endswith("/learning_rate"):
        return "Learning Rate"
    return tag.split("/")[-1].replace("_", " ")


def _plot_scalar(*, steps: list[int], values: list[float], tag: str, title: str, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9.75))
    ax.plot(steps, values, linewidth=2.8, color="#0b6e4f")
    if values:
        ax.scatter([steps[-1]], [values[-1]], color="#b22222", s=64, zorder=3)
    ax.set_xlabel("Step", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(_ylabel(tag), fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{title}\n{_pretty_metric_name(tag)}", fontsize=TITLE_FONTSIZE, pad=18)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_accuracy(
    *,
    train_steps: list[int],
    train_values: list[float],
    test_steps: list[int],
    test_values: list[float],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9.75))
    ax.plot(train_steps, train_values, linewidth=3.0, color="#0b6e4f", label="Train Accuracy")
    ax.plot(test_steps, test_values, linewidth=3.0, color="#b22222", label="Test Accuracy")
    if train_values:
        ax.scatter([train_steps[-1]], [train_values[-1]], color="#0b6e4f", s=72, zorder=3)
    if test_values:
        ax.scatter([test_steps[-1]], [test_values[-1]], color="#b22222", s=72, zorder=3)
    ax.set_xlabel("Step", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{title}\nTrain vs Test Accuracy", fontsize=TITLE_FONTSIZE, pad=18)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-replot-run",
        description="Regenerate run plots from a TensorBoard event file with a custom title.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--all-tags",
        action="store_true",
        help="Plot every scalar tag instead of only the subset already present in plots/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else (run_dir / "plots")
    event_path = _find_event_file(run_dir)

    accumulator = event_accumulator.EventAccumulator(
        str(event_path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    scalar_tags = list(accumulator.Tags().get("scalars", []))
    if not scalar_tags:
        raise ValueError(f"No scalar tags found in {event_path}.")

    existing_relpaths = _existing_plot_relpaths(output_dir)
    rendered = 0
    scalar_data: dict[str, tuple[list[int], list[float]]] = {}
    for tag in scalar_tags:
        relpath = _tag_to_relpath(tag)
        if not args.all_tags and existing_relpaths and relpath.as_posix() not in existing_relpaths:
            continue

        events = accumulator.Scalars(tag)
        if not events:
            continue
        steps = [int(item.step) for item in events]
        values = [float(item.value) for item in events]
        scalar_data[tag] = (steps, values)
        _plot_scalar(
            steps=steps,
            values=values,
            tag=tag,
            title=args.title,
            output_path=output_dir / relpath,
            dpi=args.dpi,
        )
        rendered += 1

    if "train/accuracy" in scalar_data and "val/accuracy" in scalar_data:
        train_steps, train_values = scalar_data["train/accuracy"]
        test_steps, test_values = scalar_data["val/accuracy"]
        _plot_combined_accuracy(
            train_steps=train_steps,
            train_values=train_values,
            test_steps=test_steps,
            test_values=test_values,
            title=args.title,
            output_path=output_dir / COMBINED_ACCURACY_FILE,
            dpi=args.dpi,
        )
        rendered += 1

    print(f"replot: event_file={event_path}")
    print(f"replot: output_dir={output_dir}")
    print(f"replot: rendered={rendered}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
