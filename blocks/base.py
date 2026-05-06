from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.misc import tensor_stats


@dataclass
class BlockFreeCache:
    h_prev: torch.Tensor
    drive: torch.Tensor
    free_state: list[torch.Tensor]
    output: torch.Tensor


class DigitalDRNBlock(nn.Module):
    """Shared runtime wrapper for FF-driven DRN energy blocks."""

    def __init__(
        self,
        *,
        ff: nn.Module,
        energy,
        ff_learning_rate: float | None = None,
        num_iterations: int = 6,
        mode: str = "asynchronous",
        learn_drive_scale: bool = True,
        init_drive_scale: float = 1.0,
        learn_amplification: bool = False,
        init_voltage_amp: float | None = None,
        init_current_amp: float | None = None,
        amp_learning_rate: float | None = None,
        drn_learning_rate: float | None = None,
    ) -> None:
        super().__init__()

        if init_drive_scale <= 0.0:
            raise ValueError("init_drive_scale must be strictly positive.")
        if init_voltage_amp is None:
            init_voltage_amp = float(getattr(energy, "_voltage_amp", 1.0))
        if init_current_amp is None:
            init_current_amp = float(getattr(energy, "_current_amp", 1.0))
        if init_voltage_amp <= 0.0:
            raise ValueError("init_voltage_amp must be strictly positive.")
        if init_current_amp <= 0.0:
            raise ValueError("init_current_amp must be strictly positive.")

        self.ff = ff
        self.ff_learning_rate = ff_learning_rate
        self.drn_learning_rate = drn_learning_rate
        self.amp_learning_rate = amp_learning_rate

        initial_raw_drive_scale = torch.log(torch.expm1(torch.tensor(float(init_drive_scale))))
        self._drive_scale_raw = nn.Parameter(initial_raw_drive_scale.clone(), requires_grad=learn_drive_scale)
        self.learn_amplification = bool(learn_amplification)
        if self.learn_amplification:
            initial_raw_voltage_amp = torch.log(torch.expm1(torch.tensor(float(init_voltage_amp))))
            initial_raw_current_amp = torch.log(torch.expm1(torch.tensor(float(init_current_amp))))
            self._voltage_amp_raw = nn.Parameter(initial_raw_voltage_amp.clone(), requires_grad=True)
            self._current_amp_raw = nn.Parameter(initial_raw_current_amp.clone(), requires_grad=True)
            self._fixed_voltage_amp = None
            self._fixed_current_amp = None
        else:
            self._voltage_amp_raw = None
            self._current_amp_raw = None
            self._fixed_voltage_amp = float(init_voltage_amp)
            self._fixed_current_amp = float(init_current_amp)

        self.energy = energy
        self.augmented_energy = self.energy.build_augmented_energy()
        self.inference_minimizer = self.energy.build_minimizer(
            num_iterations=num_iterations,
            mode=mode,
        )
        self.training_minimizer = self.energy.build_minimizer(
            fn=self.augmented_energy,
            num_iterations=num_iterations,
            mode=mode,
        )
        self.augmented_minimizer = self.training_minimizer
        self.minimizer = self.inference_minimizer
        self._device = None
        self._sync_amplification()

    @staticmethod
    def _canonical_device(device: torch.device | str):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def _sync_energy_device(self, device: torch.device):
        device = self._canonical_device(device)
        energy_device = self.energy._device
        if energy_device is not None:
            energy_device = self._canonical_device(energy_device)
        if energy_device != device:
            self.energy.set_device(device)
        self._device = device

    @property
    def drive_scale(self):
        return F.softplus(self._drive_scale_raw)

    @property
    def voltage_amp(self):
        if self._voltage_amp_raw is None:
            return self._fixed_voltage_amp
        return F.softplus(self._voltage_amp_raw)

    @property
    def current_amp(self):
        if self._current_amp_raw is None:
            return self._fixed_current_amp
        return F.softplus(self._current_amp_raw)

    def _sync_amplification(self):
        voltage_amp = self.voltage_amp
        current_amp = self.current_amp

        for obj in (
            self.energy,
            getattr(self, "augmented_energy", None),
            getattr(self, "inference_minimizer", None),
            getattr(self, "training_minimizer", None),
            getattr(self, "augmented_minimizer", None),
            getattr(self, "minimizer", None),
        ):
            if obj is None:
                continue
            if hasattr(obj, "_voltage_amp"):
                obj._voltage_amp = voltage_amp
            if hasattr(obj, "_current_amp"):
                obj._current_amp = current_amp
            for updater in getattr(obj, "_updaters", []):
                updater.voltage_amp = voltage_amp
                updater.current_amp = current_amp

        for interaction in getattr(self.energy, "_interactions", []):
            if hasattr(interaction, "_voltage_amp"):
                interaction._voltage_amp = voltage_amp
            if hasattr(interaction, "_current_amp"):
                interaction._current_amp = current_amp
        return voltage_amp, current_amp

    def output_state(self):
        return self.energy.output_state()

    def output_layer(self):
        return self.energy.output_layer()

    def free_layers(self):
        return self.energy.free_layers()

    def set_drive(self, h: torch.Tensor):
        self._sync_amplification()
        current = self.drive_scale * self.ff(h)
        self.energy.set_drive(current)
        return current

    def capture_free_cache(self, h_prev: torch.Tensor) -> BlockFreeCache:
        drive = self.energy.drive.current
        if drive is None:
            raise RuntimeError("Cannot capture block cache before the free-phase drive has been set.")
        return BlockFreeCache(
            h_prev=h_prev.detach().clone(),
            drive=drive.detach().clone(),
            free_state=[layer.state.detach().clone() for layer in self.free_layers()],
            output=self.output_state().detach().clone(),
        )

    def restore_free_cache(self, cache: BlockFreeCache) -> None:
        self._sync_energy_device(cache.drive.device)
        for layer, state in zip(self.free_layers(), cache.free_state):
            requires_grad = layer.state.requires_grad
            layer.state = state.detach().clone().to(device=self._device)
            layer.state.requires_grad_(requires_grad)
        self.energy.set_drive(cache.drive.detach().clone().to(device=self._device))

    def collect_diagnostics(self):
        drive = self.energy.drive.current
        if drive is None:
            raise RuntimeError("Diagnostics requested before drive current was set.")
        free_layers = self.energy.free_layers()
        first_state = free_layers[0].state
        output_state = self.output_state()

        diagnostics = {
            "drive_scale": float(self.drive_scale.detach().item()),
            "voltage_amp": _scalar(self.voltage_amp),
            "current_amp": _scalar(self.current_amp),
        }
        diagnostics.update(tensor_stats(drive, "drive"))
        diagnostics.update(tensor_stats(first_state, "z1"))
        diagnostics.update(tensor_stats(output_state, "z_out"))

        z1_rms = max(diagnostics["z1_rms"], 1.0e-12)
        diagnostics["drive_to_z1_rms_ratio"] = float(diagnostics["drive_rms"] / z1_rms)
        dense_weights = getattr(self.energy, "dense_weights", None)
        if dense_weights:
            weight = dense_weights[0].get().detach().float()
            if weight.ndim == 2:
                row_half_sum = 0.5 * weight.sum(dim=1)
                col_half_sum = 0.5 * weight.sum(dim=0)
                diagnostics.update(_conductance_stats(weight, "dense0_weight"))
                diagnostics.update(_vector_stats(row_half_sum, "dense0_a_pre"))
                diagnostics.update(_vector_stats(col_half_sum, "dense0_a_post"))
        biases = getattr(self.energy, "biases", None)
        if biases:
            diagnostics.update(_conductance_stats(biases[0].get().detach().float(), "bias0"))
        return diagnostics

    def set_device(self, device: torch.device):
        device = self._canonical_device(device)
        self.ff.to(device)
        self._sync_energy_device(device)
        self._sync_amplification()
        return self

    def resistive_params(self):
        return self.energy.params()

    def resistive_param_states(self):
        return [param.state for param in self.resistive_params()]

    def ff_parameters(self):
        params = list(self.ff.parameters())
        if self._drive_scale_raw.requires_grad:
            params.append(self._drive_scale_raw)
        return params

    def named_ff_parameters(self):
        for name, param in self.ff.named_parameters():
            yield name, param
        if self._drive_scale_raw.requires_grad:
            yield "drive_scale_raw", self._drive_scale_raw

    def amplification_parameters(self):
        params = []
        if self._voltage_amp_raw is not None and self._voltage_amp_raw.requires_grad:
            params.append(self._voltage_amp_raw)
        if self._current_amp_raw is not None and self._current_amp_raw.requires_grad:
            params.append(self._current_amp_raw)
        return params

    def named_amplification_parameters(self):
        if self._voltage_amp_raw is not None and self._voltage_amp_raw.requires_grad:
            yield "voltage_amp_raw", self._voltage_amp_raw
        if self._current_amp_raw is not None and self._current_amp_raw.requires_grad:
            yield "current_amp_raw", self._current_amp_raw

    def named_resistive_parameters(self):
        for param in self.resistive_params():
            name = getattr(param, "name", param.__class__.__name__)
            yield name, param.state

    def optimizer_param_groups(self):
        groups = []
        if hasattr(self.ff, "optimizer_param_groups"):
            for group in self.ff.optimizer_param_groups():
                params = list(group.get("params", []))
                if not params:
                    continue
                normalized_group = dict(group)
                normalized_group["params"] = params
                if self.ff_learning_rate is not None and "lr" not in normalized_group:
                    normalized_group["lr"] = self.ff_learning_rate
                groups.append(normalized_group)
            if self._drive_scale_raw.requires_grad:
                group = {"params": [self._drive_scale_raw]}
                if self.ff_learning_rate is not None:
                    group["lr"] = self.ff_learning_rate
                groups.append(group)
        else:
            ff_params = list(self.ff.parameters())
            if self._drive_scale_raw.requires_grad:
                ff_params.append(self._drive_scale_raw)
            if ff_params:
                group = {"params": ff_params}
                if self.ff_learning_rate is not None:
                    group["lr"] = self.ff_learning_rate
                groups.append(group)

        amp_params = self.amplification_parameters()
        if amp_params:
            group = {"params": amp_params}
            if self.amp_learning_rate is not None:
                group["lr"] = self.amp_learning_rate
            groups.append(group)

        drn_params = self.resistive_param_states()
        if drn_params:
            group = {"params": drn_params}
            if self.drn_learning_rate is not None:
                group["lr"] = self.drn_learning_rate
            groups.append(group)
        return groups

    def enable_resistive_grad_(self, enabled: bool = True):
        for param in self.resistive_params():
            param.state.requires_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True):
        self.energy.zero_param_grad(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self):
        self.energy.clamp_params_()
        return self

    def detach_state_(self):
        self.energy.detach_state()
        return self

    def reset_state(self, batch_size: int, device: torch.device):
        self._sync_energy_device(device)
        self.energy.reset_free_layers(batch_size, device=device)

    def equilibrate(self, num_iterations: int | None = None):
        self._sync_amplification()
        if num_iterations is None:
            self.minimizer.compute_equilibrium()
            return

        original_iterations = self.minimizer.num_iterations
        self.minimizer.num_iterations = num_iterations
        try:
            self.minimizer.compute_equilibrium()
        finally:
            self.minimizer.num_iterations = original_iterations

    def forward(self, h: torch.Tensor, reset: bool = False, num_iterations: int | None = None):
        self._sync_energy_device(h.device)
        self._sync_amplification()

        needs_reset = (
            reset
            or self.output_state().size(0) != h.size(0)
            or self.output_state().device != h.device
        )
        if needs_reset:
            self.reset_state(h.size(0), h.device)

        self.set_drive(h)
        self.equilibrate(num_iterations=num_iterations)
        return self.output_state()


def _conductance_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    finite_tensor = tensor[finite]
    if finite_tensor.numel() == 0:
        return {
            f"{prefix}_finite_frac": 0.0,
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_zero_frac": float("nan"),
        }
    return {
        f"{prefix}_finite_frac": float(finite.float().mean().item()),
        f"{prefix}_min": float(finite_tensor.min().item()),
        f"{prefix}_max": float(finite_tensor.max().item()),
        f"{prefix}_rms": float(torch.sqrt(torch.mean(finite_tensor * finite_tensor)).item()),
        f"{prefix}_zero_frac": float((finite_tensor == 0.0).float().mean().item()),
    }


def _vector_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    finite_tensor = tensor[finite]
    if finite_tensor.numel() == 0:
        return {
            f"{prefix}_finite_frac": 0.0,
            f"{prefix}_min": float("nan"),
            f"{prefix}_q001": float("nan"),
            f"{prefix}_median": float("nan"),
        }
    return {
        f"{prefix}_finite_frac": float(finite.float().mean().item()),
        f"{prefix}_min": float(finite_tensor.min().item()),
        f"{prefix}_q001": float(torch.quantile(finite_tensor, 0.001).item()),
        f"{prefix}_median": float(torch.quantile(finite_tensor, 0.5).item()),
    }


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)
