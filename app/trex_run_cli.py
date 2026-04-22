from __future__ import annotations

import argparse
import shlex

from .cli import main as train_main
from .trex_common import local_config_path, normalize_training_args


def _forwarded_train_args(
    *,
    config_name: str,
    config_dir: str | None,
    training_args: list[str],
) -> list[str]:
    normalized = normalize_training_args(training_args)
    reserved = {"--config-name", "--config-dir"}
    duplicates = [arg for arg in normalized if arg in reserved]
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Pass {joined} to digital-drn-trex-run directly, not through the remainder args.")

    forwarded: list[str] = ["--config-name", config_name]
    if config_dir is not None:
        forwarded.extend(["--config-dir", config_dir])
    forwarded.extend(normalized)
    return forwarded


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-trex-run",
        description="Run a previously copied Hydra config on the local machine.",
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    local_config_path(args.config_name, args.config_dir)
    forwarded = _forwarded_train_args(
        config_name=args.config_name,
        config_dir=args.config_dir,
        training_args=list(args.training_args),
    )

    print(f"+ digital-drn-train {shlex.join(forwarded)}")
    if args.dry_run:
        return 0
    return train_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
