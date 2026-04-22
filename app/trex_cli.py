from __future__ import annotations

import argparse
import shlex
import subprocess
from typing import Sequence

from .trex_common import build_target, local_config_path, remote_config_path


def _run_or_print(command: Sequence[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}")
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-trex-scp",
        description="Copy a Hydra config to a remote host.",
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--host", default="trex")
    parser.add_argument("--user", default="filip")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--remote-root", default="/home/filip/digital_drn")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    target = build_target(args.user, args.host)
    config_path = local_config_path(args.config_name, args.config_dir)
    destination_path = remote_config_path(args.config_name, args.remote_root)

    mkdir_command = [
        "ssh",
        target,
        f"mkdir -p {shlex.quote(str(destination_path.parent))}",
    ]
    scp_command = [
        "scp",
        str(config_path),
        f"{target}:{destination_path}",
    ]

    _run_or_print(mkdir_command, dry_run=args.dry_run)
    _run_or_print(scp_command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
