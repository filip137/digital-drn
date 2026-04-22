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
        drn_learning_rate: float | None = None,
    ) -> None:
        super().__init__()

        if init_drive_scale <= 0.0:
            raise ValueError("init_drive_scale must be strictly positive.")

        self.ff = ff
        self.ff_learning_rate = ff_learning_rate
        self.drn_learning_rate = drn_learning_rate

        initial_raw_drive_scale = torch.log(torch.expm1(torch.tensor(float(init_drive_scale))))
        self._drive_scale_raw = nn.Parameter(initial_raw_drive_scale.clone(), requires_grad=learn_drive_scale)

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

    def output_state(self):
        return self.energy.output_state()

    def output_layer(self):
        return self.energy.output_layer()

    def free_layers(self):
        return self.energy.free_layers()

    def set_drive(self, h: torch.Tensor):
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

        diagnostics = {"drive_scale": float(self.drive_scale.detach().item())}
        diagnostics.update(tensor_stats(drive, "drive"))
        diagnostics.update(tensor_stats(first_state, "z1"))
        diagnostics.update(tensor_stats(output_state, "z_out"))

        z1_rms = max(diagnostics["z1_rms"], 1.0e-12)
        diagnostics["drive_to_z1_rms_ratio"] = float(diagnostics["drive_rms"] / z1_rms)
        return diagnostics

    def set_device(self, device: torch.device):
        device = self._canonical_device(device)
        self.ff.to(device)
        self._sync_energy_device(device)
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
