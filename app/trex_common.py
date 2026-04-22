from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Sequence


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_target(user: str | None, host: str) -> str:
    if user is None or user == "":
        return host
    return f"{user}@{host}"


def local_config_path(config_name: str, config_dir: str | Path | None = None) -> Path:
    base_dir = Path(config_dir) if config_dir is not None else package_root() / "hydra_conf"
    path = base_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path.resolve()


def remote_config_path(config_name: str, remote_root: str) -> PurePosixPath:
    return PurePosixPath(remote_root) / "hydra_conf" / f"{config_name}.yaml"


def normalize_training_args(training_args: Sequence[str]) -> list[str]:
    normalized = list(training_args)
    if normalized[:1] == ["--"]:
        normalized = normalized[1:]
    return normalized
