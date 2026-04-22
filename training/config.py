from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1.0e-4
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    epochs: int = 20
    criterion: str = "cross_entropy"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    grad_clip_norm: float | None = None
    log_every: int = 50
    eval_every: int = 1
    save_every: int = 0
    save_best: bool = True
    save_last: bool = False
    checkpoint_dir: str | Path | None = None
    save_history: bool = True
    save_events: bool = True
    device: str = "auto"
    seed: int | None = 0
    deterministic: bool = False
    train_num_iterations: int | None = None
    eval_num_iterations: int | None = None
    train_reset_state: bool = True
    eval_reset_state: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.checkpoint_dir is not None:
            data["checkpoint_dir"] = str(self.checkpoint_dir)
        return data
