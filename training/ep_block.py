from __future__ import annotations

from dataclasses import dataclass

import torch

from ..blocks.base import BlockFreeCache, DigitalDRNBlock
from ..core.parameter import Bias


def _layer_index(layer) -> int:
    name = getattr(layer, "name", getattr(layer, "_name", layer))
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Expected layer name ending in '_<index>', got {name!r}.") from exc


@dataclass
class BlockEPResult:
    param_grads: list[torch.Tensor]
    delta_drive: torch.Tensor
    plus_state: list[torch.Tensor]
    minus_state: list[torch.Tensor]
    plus_output: torch.Tensor
    minus_output: torch.Tensor


class BlockEquilibriumProp:
    """Centered EP helper for one DRN block with fixed free-phase drive."""

    def __init__(
        self,
        block: DigitalDRNBlock,
        beta: float,
        *,
        amp_gradient_compensation: bool = False,
        downstream_weight_count: int = 0,
        num_iterations: int | None = None,
    ) -> None:
        if beta <= 0.0:
            raise ValueError("beta must be strictly positive for centered EP.")
        if downstream_weight_count < 0:
            raise ValueError("downstream_weight_count must be non-negative.")
        if num_iterations is not None and num_iterations <= 0:
            raise ValueError("num_iterations must be positive when provided.")
        self.block = block
        self.beta = float(beta)
        self.amp_gradient_compensation = bool(amp_gradient_compensation)
        self.downstream_weight_count = int(downstream_weight_count)
        self.num_iterations = None if num_iterations is None else int(num_iterations)

    def _restore_state_and_drive(self, state_cache: list[torch.Tensor], drive: torch.Tensor) -> None:
        self.block._sync_energy_device(drive.device)
        for layer, state in zip(self.block.free_layers(), state_cache):
            requires_grad = layer.state.requires_grad
            layer.state = state.detach().clone().to(device=self.block._device)
            layer.state.requires_grad_(requires_grad)
        self.block.energy.set_drive(drive.detach().clone().to(device=self.block._device))

    def _energy_param_grads(self, state_cache: list[torch.Tensor], drive: torch.Tensor) -> list[torch.Tensor]:
        self._restore_state_and_drive(state_cache, drive)
        batch_size = float(state_cache[0].shape[0])
        # The nudging force already carries the loss reduction. Resistive
        # parameter gradients therefore need the batch-summed energy here,
        # not an extra batch mean. We use the existing efficient param-grad
        # helpers and rescale by batch size instead of materializing the full
        # summed conv energy, which can be prohibitively large.
        return [
            (self.block.energy.grad_param_fn(param)() * batch_size).detach().clone()
            for param in self.block.resistive_params()
        ]

    def _amp_compensation_scales(self) -> tuple[list[float], float]:
        if not self.amp_gradient_compensation:
            return [1.0] * len(self.block.resistive_params()), 1.0

        voltage_amp = float(getattr(self.block.energy, "_voltage_amp", 1.0))
        current_amp = float(getattr(self.block.energy, "_current_amp", 1.0))
        if abs(voltage_amp - current_amp) <= 1.0e-12:
            return [1.0] * len(self.block.resistive_params()), 1.0
        if abs(current_amp) <= 1.0e-12:
            raise ValueError("ad-hoc EP amplitude compensation requires non-zero current_amp.")

        weight_count = sum(0 if isinstance(param, Bias) else 1 for param in self.block.resistive_params())
        ratio = float(voltage_amp / current_amp)
        base_exponent = self.downstream_weight_count + weight_count

        scales: list[float] = []
        bias_index = 0
        for param in self.block.resistive_params():
            if isinstance(param, Bias):
                exponent = self.downstream_weight_count + (weight_count - bias_index)
                bias_index += 1
            else:
                exponent = base_exponent
            scales.append(ratio**exponent)
        return scales, ratio**base_exponent

    def _amplified_current_bias_gradient_scale(self, param) -> float:
        if not getattr(self.block.augmented_energy, "amplified_current_correction_enabled", False):
            return 1.0
        if not isinstance(param, Bias):
            return 1.0

        voltage_amp = getattr(self.block.energy, "_voltage_amp", None)
        current_amp = getattr(self.block.energy, "_current_amp", None)
        if voltage_amp in (None, 0.0) or current_amp is None:
            return 1.0

        for interaction in getattr(self.block.energy, "_interactions", []):
            if getattr(interaction, "_bias", None) is param:
                layer = getattr(interaction, "_layer", None)
                if layer is None:
                    return 1.0
                return float(current_amp / voltage_amp) ** max(_layer_index(layer) - 1, 0)
        return 1.0

    def _capture_state_and_output(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        return (
            [layer.state.detach().clone() for layer in self.block.free_layers()],
            self.block.output_state().detach().clone(),
        )

    def _compute_training_equilibrium(self) -> None:
        if self.num_iterations is None:
            self.block.training_minimizer.compute_equilibrium()
            return

        original_iterations = self.block.training_minimizer.num_iterations
        self.block.training_minimizer.num_iterations = self.num_iterations
        try:
            self.block.training_minimizer.compute_equilibrium()
        finally:
            self.block.training_minimizer.num_iterations = original_iterations

    def compute_gradients(
        self,
        *,
        free_cache: BlockFreeCache,
        output_cotangent: torch.Tensor,
    ) -> BlockEPResult:
        self.block.augmented_energy.prepare_nudging(
            output_gradient=output_cotangent.detach(),
            nudging=self.beta,
        )

        self.block.restore_free_cache(free_cache)
        self.block.augmented_energy.nudging.nudging = self.beta
        self._compute_training_equilibrium()
        plus_state, plus_output = self._capture_state_and_output()

        self.block.restore_free_cache(free_cache)
        self.block.augmented_energy.nudging.nudging = -self.beta
        self._compute_training_equilibrium()
        minus_state, minus_output = self._capture_state_and_output()

        plus_param_grads = self._energy_param_grads(plus_state, free_cache.drive)
        minus_param_grads = self._energy_param_grads(minus_state, free_cache.drive)
        param_scales, drive_scale = self._amp_compensation_scales()
        param_grads = [
            ((grad_plus - grad_minus) / (2.0 * self.beta))
            * scale
            * self._amplified_current_bias_gradient_scale(param)
            for param, grad_plus, grad_minus, scale in zip(
                self.block.resistive_params(),
                plus_param_grads,
                minus_param_grads,
                param_scales,
            )
        ]

        # The current block energies use linear drive coupling -<z1, x>, so
        # ∂E/∂x = -z1 and the centered cotangent is (z1^- - z1^+) / (2β).
        delta_drive = ((minus_state[0] - plus_state[0]) / (2.0 * self.beta)) * drive_scale

        self.block.restore_free_cache(free_cache)
        return BlockEPResult(
            param_grads=param_grads,
            delta_drive=delta_drive.detach().clone(),
            plus_state=plus_state,
            minus_state=minus_state,
            plus_output=plus_output,
            minus_output=minus_output,
        )
