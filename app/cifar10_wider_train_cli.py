from __future__ import annotations

import argparse
import shlex

from .cli import main as train_main
from .trex_common import local_config_path, normalize_training_args


DEFAULT_CONFIG_NAME = "cifar10_drn_only_signed_norm_readout_wider"


def _forwarded_train_args(
    *,
    config_dir: str | None,
    training_args: list[str],
) -> list[str]:
    normalized = normalize_training_args(training_args)
    reserved = {"--config-name", "--config-dir"}
    duplicates = [arg for arg in normalized if arg in reserved]
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Pass {joined} to digital-drn-cifar10-wider-train directly, not through the remainder args.")

    forwarded: list[str] = ["--config-name", DEFAULT_CONFIG_NAME]
    if config_dir is not None:
        forwarded.extend(["--config-dir", config_dir])
    forwarded.extend(normalized)
    return forwarded


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-cifar10-wider-train",
        description="Train the widened CIFAR-10 signed-readout config with the repo's standard train CLI.",
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    local_config_path(DEFAULT_CONFIG_NAME, args.config_dir)
    forwarded = _forwarded_train_args(
        config_dir=args.config_dir,
        training_args=list(args.training_args),
    )

    print(f"+ digital-drn-train {shlex.join(forwarded)}")
    if args.dry_run:
        return 0
    return train_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
