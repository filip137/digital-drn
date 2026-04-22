from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
try:  # pragma: no cover - tensorboard may be optional in minimal installs
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None
from torch.utils.data import DataLoader

try:  # pragma: no cover - numpy is expected but optional in minimal installs
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .config import OptimizerConfig, SchedulerConfig, TrainerConfig
from .ep_network import hybrid_backward_explicit
from ..models.network import DigitalDRNNet
from ..models.network_digital_analog import DigitalAnalogNet
from ..utils.misc import resolve_device, set_seed


class NormalizedMSELoss(nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.mse_loss(input, target, reduction=self.reduction)


_CRITERIA = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": NormalizedMSELoss,
}

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

_SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multi_step": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cyclic": torch.optim.lr_scheduler.CyclicLR,
    "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    num_samples: int
    num_batches: int


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epochs_completed: int = 0
    steps_completed: int = 0
    best_val_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "epochs_completed": self.epochs_completed,
            "steps_completed": self.steps_completed,
            "best_val_accuracy": self.best_val_accuracy,
        }


def build_criterion(name: str) -> nn.Module:
    if name not in _CRITERIA:
        raise ValueError(f"Unsupported criterion '{name}'.")
    if name == "cross_entropy":
        return _CRITERIA[name]()
    return _CRITERIA[name](reduction="mean")


def build_optimizer(model: DigitalDRNNet, cfg: OptimizerConfig) -> Optimizer:
    if cfg.name not in _OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{cfg.name}'.")
    Optim = _OPTIMIZERS[cfg.name]
    param_groups = model.optimizer_param_groups()
    if param_groups:
        return Optim(param_groups, lr=cfg.lr, **cfg.config)
    return Optim(model.optimizer_tensors(), lr=cfg.lr, **cfg.config)


def build_scheduler(optimizer: Optimizer, cfg: SchedulerConfig | None) -> LRScheduler | None:
    if cfg is None or cfg.name is None:
        return None
    if cfg.name not in _SCHEDULERS:
        raise ValueError(f"Unsupported scheduler '{cfg.name}'.")
    Scheduler = _SCHEDULERS[cfg.name]
    return Scheduler(optimizer, **cfg.config)


class BPTrainer:
    @staticmethod
    def _normalize_rng_tensor(state: Any) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.detach().to(device="cpu", dtype=torch.uint8)
        if np is not None and isinstance(state, np.ndarray):
            return torch.as_tensor(state, dtype=torch.uint8, device="cpu")
        return torch.as_tensor(state, dtype=torch.uint8, device="cpu")

    def __init__(
        self,
        model: DigitalDRNNet,
        config: TrainerConfig | None = None,
        *,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        experiment_config: Mapping[str, Any] | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.experiment_config = dict(experiment_config) if experiment_config is not None else None
        self.run_metadata = dict(run_metadata) if run_metadata is not None else None

        if self.config.seed is not None:
            set_seed(self.config.seed, deterministic=self.config.deterministic)

        self.device = resolve_device(self.config.device)
        self.model.set_device(self.device)
        self.model.enable_resistive_grad_()

        self.criterion = (criterion or build_criterion(self.config.criterion)).to(self.device)
        self.optimizer = optimizer or build_optimizer(self.model, self.config.optimizer)
        self.scheduler = scheduler if scheduler is not None else build_scheduler(self.optimizer, self.config.scheduler)

        self.history = TrainHistory()
        self.checkpoint_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir is not None else None
        self.writer = None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if self.config.save_events and SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir))
            self._save_config_snapshot()
            self._save_experiment_snapshot()
            self._save_run_metadata()

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _move_batch(batch, device: torch.device):
        if len(batch) < 2:
            raise ValueError("Expected batches to contain at least (inputs, targets).")
        x, y = batch[:2]
        return x.to(device), y.to(device)

    @staticmethod
    def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        if logits.ndim < 2 or targets.ndim != 1:
            return 0.0
        return float((logits.argmax(dim=1) == targets).sum().item()) / max(int(targets.numel()), 1)

    def _maybe_clip_gradients(self) -> None:
        if self.config.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(self.model.optimizer_tensors(), self.config.grad_clip_norm)

    @staticmethod
    def _sanitize_tag_component(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return sanitized.replace(".", "_").strip("_") or "param"

    def _write_tensor_stats(self, base_tag: str, tensor: torch.Tensor, global_step: int) -> None:
        stats = {
            "mean": float(tensor.detach().float().mean().item()),
            "std": float(tensor.detach().float().std(unbiased=False).item()),
            "rms": float(tensor.detach().float().square().mean().sqrt().item()),
            "max_abs": float(tensor.detach().float().abs().max().item()),
        }
        for key, value in stats.items():
            self.writer.add_scalar(f"{base_tag}/{key}", value, global_step)

    def _log_parameter_summaries(self, global_step: int) -> None:
        if self.writer is None:
            return

        for block_idx, block in enumerate(self.model.blocks):
            if hasattr(block, "named_ff_parameters"):
                for name, param in block.named_ff_parameters():
                    safe_name = self._sanitize_tag_component(name)
                    self._write_tensor_stats(f"weights/block_{block_idx}/ff/{safe_name}", param, global_step)
                    if param.grad is not None:
                        self._write_tensor_stats(f"gradients/block_{block_idx}/ff/{safe_name}", param.grad, global_step)

            if hasattr(block, "named_resistive_parameters"):
                for name, tensor in block.named_resistive_parameters():
                    safe_name = self._sanitize_tag_component(name)
                    self._write_tensor_stats(f"weights/block_{block_idx}/drn/{safe_name}", tensor, global_step)
                    if tensor.grad is not None:
                        self._write_tensor_stats(f"gradients/block_{block_idx}/drn/{safe_name}", tensor.grad, global_step)

        head = getattr(self.model, "head", None)
        if head is not None:
            for name, param in head.named_parameters():
                safe_name = self._sanitize_tag_component(name)
                self._write_tensor_stats(f"weights/head/{safe_name}", param, global_step)
                if param.grad is not None:
                    self._write_tensor_stats(f"gradients/head/{safe_name}", param.grad, global_step)

    def _log_probe_diagnostics(
        self,
        dataloader: DataLoader | None,
        *,
        split: str,
        global_step: int,
        num_iterations: int | None,
        reset_state: bool,
    ) -> None:
        if self.writer is None or dataloader is None:
            return

        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                inputs, _targets = self._move_batch(batch, self.device)
                self.model(inputs, reset=reset_state, num_iterations=num_iterations)
                for block_idx, block in enumerate(self.model.blocks):
                    if not hasattr(block, "collect_diagnostics"):
                        continue
                    diagnostics = block.collect_diagnostics()
                    for key, value in diagnostics.items():
                        self.writer.add_scalar(
                            f"diagnostics/{split}/block_{block_idx}/{key}",
                            value,
                            global_step,
                        )
                self.model.detach_state_()
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch_index: int = 0,
        max_steps: int | None = None,
    ) -> EpochMetrics:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch in dataloader:
            if max_steps is not None and self.history.steps_completed >= max_steps:
                break

            inputs, targets = self._move_batch(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(
                inputs,
                reset=self.config.train_reset_state,
                num_iterations=self.config.train_num_iterations,
            )
            loss = self.criterion(logits, targets)
            loss.backward()
            self._maybe_clip_gradients()
            self.optimizer.step()
            self.model.clamp_resistive_params_()
            self.model.detach_state_()

            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_samples += batch_size
            num_batches += 1
            self.history.steps_completed += 1

            if self.config.log_every and (num_batches % self.config.log_every == 0):
                avg_loss = total_loss / max(total_samples, 1)
                avg_acc = total_correct / max(total_samples, 1)
                print(
                    f"epoch {epoch_index + 1} step {self.history.steps_completed}: "
                    f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}"
                )

        return EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            num_samples=total_samples,
            num_batches=num_batches,
        )

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        max_steps: int | None = None,
    ) -> EpochMetrics:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_steps is not None and batch_idx >= max_steps:
                    break
                inputs, targets = self._move_batch(batch, self.device)
                logits = self.model(
                    inputs,
                    reset=self.config.eval_reset_state,
                    num_iterations=self.config.eval_num_iterations,
                )
                loss = self.criterion(logits, targets)
                self.model.detach_state_()

                batch_size = int(targets.size(0))
                total_loss += float(loss.item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                total_samples += batch_size
                num_batches += 1

        return EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            num_samples=total_samples,
            num_batches=num_batches,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        max_steps: int | None = None,
    ) -> TrainHistory:
        best_metric = self.history.best_val_accuracy

        for epoch in range(self.history.epochs_completed, self.config.epochs):
            train_metrics = self.train_epoch(train_loader, epoch_index=epoch, max_steps=max_steps)
            self.history.train_loss.append(train_metrics.loss)
            self.history.train_accuracy.append(train_metrics.accuracy)
            self.history.learning_rate.append(self._current_lr())
            self.history.epochs_completed = epoch + 1

            val_metrics = None
            if val_loader is not None and self.config.eval_every and ((epoch + 1) % self.config.eval_every == 0):
                val_metrics = self.evaluate(val_loader)
                self.history.val_loss.append(val_metrics.loss)
                self.history.val_accuracy.append(val_metrics.accuracy)
                metric = val_metrics.accuracy
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    self.history.best_val_accuracy = metric
                    if self.checkpoint_dir is not None and self.config.save_best:
                        self.save_checkpoint(self.checkpoint_dir / "checkpoint_best.pt")

            if self.scheduler is not None:
                self.scheduler.step()

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics.loss, epoch + 1)
                self.writer.add_scalar("train/accuracy", train_metrics.accuracy, epoch + 1)
                self.writer.add_scalar("train/learning_rate", self._current_lr(), epoch + 1)
                if val_metrics is not None:
                    self.writer.add_scalar("val/loss", val_metrics.loss, epoch + 1)
                    self.writer.add_scalar("val/accuracy", val_metrics.accuracy, epoch + 1)
                probe_loader = val_loader if val_loader is not None else train_loader
                probe_split = "val" if val_loader is not None else "train"
                probe_iterations = self.config.eval_num_iterations if val_loader is not None else self.config.train_num_iterations
                probe_reset = self.config.eval_reset_state if val_loader is not None else self.config.train_reset_state
                self._log_probe_diagnostics(
                    probe_loader,
                    split=probe_split,
                    global_step=epoch + 1,
                    num_iterations=probe_iterations,
                    reset_state=probe_reset,
                )
                self._log_parameter_summaries(global_step=epoch + 1)
                self.writer.flush()

            print(
                f"epoch {epoch + 1}: train_loss={train_metrics.loss:.4f} "
                f"train_acc={train_metrics.accuracy:.4f}"
                + (
                    f" val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
                    if val_metrics is not None
                    else ""
                )
            )

            if self.checkpoint_dir is not None:
                if self.config.save_every and ((epoch + 1) % self.config.save_every == 0):
                    self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
                if self.config.save_history:
                    self._save_history()

            if max_steps is not None and self.history.steps_completed >= max_steps:
                break

        if self.checkpoint_dir is not None and self.config.save_last:
            self.save_checkpoint(self.checkpoint_dir / "checkpoint_last.pt")

        if self.writer is not None:
            self.writer.flush()

        return self.history

    def _save_config_snapshot(self) -> None:
        if self.checkpoint_dir is None:
            return
        config_path = self.checkpoint_dir / "trainer_config.json"
        with config_path.open("w") as handle:
            json.dump(self.config.to_dict(), handle, indent=2, default=str)

    def _save_history(self) -> None:
        if self.checkpoint_dir is None:
            return
        history_path = self.checkpoint_dir / "history.json"
        with history_path.open("w") as handle:
            json.dump(self.history.to_dict(), handle, indent=2)

    def _save_experiment_snapshot(self) -> None:
        if self.checkpoint_dir is None or self.experiment_config is None:
            return
        config_path = self.checkpoint_dir / "experiment_config.json"
        with config_path.open("w") as handle:
            json.dump(self.experiment_config, handle, indent=2, default=str)

    def _save_run_metadata(self) -> None:
        if self.checkpoint_dir is None or self.run_metadata is None:
            return
        metadata_path = self.checkpoint_dir / "run_metadata.json"
        with metadata_path.open("w") as handle:
            json.dump(self.run_metadata, handle, indent=2, default=str)

    def _optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _get_resistive_checkpoint_state(self) -> list[dict[str, Any]]:
        return [
            {
                "name": getattr(param, "name", f"param_{idx}"),
                "state": param.state.detach().cpu(),
            }
            for idx, param in enumerate(self.model.resistive_params())
        ]

    def _load_resistive_checkpoint_state(self, states: list[dict[str, Any]]) -> None:
        params = self.model.resistive_params()
        if len(states) != len(params):
            raise ValueError(
                f"Checkpoint contains {len(states)} resistive tensors but model exposes {len(params)} resistive parameters."
            )
        with torch.no_grad():
            for param, saved in zip(params, states):
                tensor = saved["state"] if isinstance(saved, dict) else saved
                param.state.copy_(tensor.to(device=param.state.device, dtype=param.state.dtype))
                param.clamp_()

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if np is not None:
            state["numpy"] = np.random.get_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(BPTrainer._normalize_rng_tensor(state["torch"]))
        if np is not None and "numpy" in state:
            np.random.set_state(state["numpy"])
        if torch.cuda.is_available() and "cuda" in state:
            cuda_states = [BPTrainer._normalize_rng_tensor(rng_state) for rng_state in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "resistive_state": self._get_resistive_checkpoint_state(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "trainer_config": self.config.to_dict(),
            "history": self.history.to_dict(),
            "rng_state": self._capture_rng_state(),
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._load_resistive_checkpoint_state(checkpoint["resistive_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._optimizer_state_to_device()
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint.get("history")
        if history:
            self.history = TrainHistory(**history)
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            self._restore_rng_state(rng_state)
        return checkpoint

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class HybridEPTrainer(BPTrainer):
    def __init__(
        self,
        model: DigitalDRNNet,
        config: TrainerConfig | None = None,
        *,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        experiment_config: Mapping[str, Any] | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        beta: float = 1.0e-3,
        nudging_mode: str = "current",
        amp_gradient_compensation: bool = False,
    ) -> None:
        if not isinstance(model, DigitalAnalogNet):
            raise TypeError("HybridEPTrainer currently expects a DigitalAnalogNet model.")
        if beta <= 0.0:
            raise ValueError("EP beta must be strictly positive.")
        if nudging_mode != "current":
            raise NotImplementedError("HybridEPTrainer currently supports only current-force nudging.")
        self.beta = float(beta)
        self.nudging_mode = str(nudging_mode)
        self.amp_gradient_compensation = bool(amp_gradient_compensation)
        super().__init__(
            model,
            config,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment_config=experiment_config,
            run_metadata=run_metadata,
        )

    @property
    def digital_analog_model(self) -> DigitalAnalogNet:
        return self.model  # type: ignore[return-value]

    @staticmethod
    def _assign_grad(param: torch.Tensor, grad: torch.Tensor | None) -> None:
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone()

    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch_index: int = 0,
        max_steps: int | None = None,
    ) -> EpochMetrics:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch in dataloader:
            if max_steps is not None and self.history.steps_completed >= max_steps:
                break

            inputs, targets = self._move_batch(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            self.model.zero_resistive_grad_(set_to_none=True)

            hybrid = hybrid_backward_explicit(
                self.digital_analog_model,
                inputs,
                targets,
                criterion=self.criterion,
                beta=self.beta,
                amp_gradient_compensation=self.amp_gradient_compensation,
                reset=self.config.train_reset_state,
                num_iterations=self.config.train_num_iterations,
            )

            for param, grad in zip(hybrid.head.head_params, hybrid.head.head_param_grads):
                self._assign_grad(param, grad)

            for block, block_result in zip(self.model.blocks, hybrid.blocks):
                for param, grad in zip(block_result.digital.ff_params, block_result.digital.ff_param_grads):
                    self._assign_grad(param, grad)
                if getattr(block, "_drive_scale_raw", None) is not None and block._drive_scale_raw.requires_grad:
                    self._assign_grad(block._drive_scale_raw, block_result.digital.drive_scale_grad)
                for resistive_param, grad in zip(block.resistive_params(), block_result.ep.param_grads):
                    self._assign_grad(resistive_param.state, grad)

            logits = hybrid.free_cache.logits.detach()
            loss = self.criterion(logits, targets).detach()

            self._maybe_clip_gradients()
            self.optimizer.step()
            self.model.clamp_resistive_params_()
            self.model.detach_state_()

            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_samples += batch_size
            num_batches += 1
            self.history.steps_completed += 1

            if self.config.log_every and (num_batches % self.config.log_every == 0):
                avg_loss = total_loss / max(total_samples, 1)
                avg_acc = total_correct / max(total_samples, 1)
                print(
                    f"epoch {epoch_index + 1} step {self.history.steps_completed}: "
                    f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}"
                )

        return EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=total_correct / max(total_samples, 1),
            num_samples=total_samples,
            num_batches=num_batches,
        )
