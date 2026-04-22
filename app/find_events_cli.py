from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_root() -> Path:
    return _package_root() / "simulation_results"


def _collect_event_files(output_root: Path, *, contains: list[str]) -> list[Path]:
    if not output_root.exists():
        return []

    event_files = [
        path
        for path in output_root.rglob("events.out.tfevents*")
        if path.is_file() and all(token in str(path) for token in contains)
    ]
    return sorted(event_files, key=lambda path: path.stat().st_mtime, reverse=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-find-events",
        description="List the newest TensorBoard event files under simulation_results.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Require this substring to appear in the path. Repeat to add more filters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root is not None else _default_output_root()
    event_files = _collect_event_files(output_root, contains=list(args.contains))

    if not event_files:
        print(f"no event files found under {output_root}")
        return 1

    for path in event_files[: args.limit]:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mtime} {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
