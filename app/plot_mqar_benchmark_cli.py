from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of history rows in {path}.")
    return data


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        if key not in row:
            raise ValueError(f"History row is missing '{key}': {row}")
        values.append(float(row[key]))
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot-mqar-benchmark",
        description="Plot MQAR benchmark histories against normalized epoch.",
    )
    parser.add_argument(
        "--history",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="History label and path to history.json. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="MQAR benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = [(label, _load_history(Path(path).expanduser())) for label, path in args.history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, rows in histories:
        epochs = _series(rows, "epoch")
        axes[0].plot(epochs, _series(rows, "train_loss"), label=f"{label} train")
        axes[0].plot(epochs, _series(rows, "val_loss"), linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, _series(rows, "val_query_acc"), label=f"{label} query")
        axes[1].plot(epochs, _series(rows, "val_exact_acc"), linestyle="--", label=f"{label} exact")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(args.title)
    out_path = out_dir / "mqar_loss_accuracy_vs_epoch.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
