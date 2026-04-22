from __future__ import annotations

from typing import Any

import torch

from ..core.interaction import LFunction, SumSeparableFunction


def _layer_index(layer) -> int:
    name = getattr(layer, "name", getattr(layer, "_name", layer))
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Expected layer name ending in '_<index>', got {name!r}.") from exc


def _amplified_layer_row_scale(energy_fn, layer) -> float:
    voltage_amp = getattr(energy_fn, "_voltage_amp", getattr(energy_fn, "voltage_amp", None))
    current_amp = getattr(energy_fn, "_current_amp", getattr(energy_fn, "current_amp", None))
    if voltage_amp in (None, 0.0) or current_amp is None:
        return 1.0
    return float(voltage_amp / current_amp) ** max(_layer_index(layer) - 1, 0)


def _resolve_current_scale(energy_fn, output_layer, mode: str, current_scale) -> float:
    if mode != "current":
        return 1.0
    if current_scale is None:
        return 1.0
    if isinstance(current_scale, str):
        if current_scale == "auto":
            return _amplified_layer_row_scale(energy_fn, output_layer)
        if current_scale in ("none", "legacy"):
            return 1.0
    return float(current_scale)


class Nudging(LFunction):
    """Current-force nudging on a block output layer."""

    def __init__(
        self,
        layer,
        *,
        nudging: float = 0.0,
        mode: str = "current",
        current_scale: float = 1.0,
    ):
        self._layer = layer
        self._nudging = float(nudging)
        self._mode = str(mode)
        self._current_scale = float(current_scale)
        self._force = None
        if self._mode != "current":
            raise NotImplementedError(
                f"Unsupported nudging mode '{self._mode}' in digital_drn. "
                "Only the current-force variant is implemented."
            )
        super().__init__([layer], [])

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def nudging(self) -> float:
        return self._nudging

    @property
    def current_scale(self) -> float:
        return self._current_scale

    @nudging.setter
    def nudging(self, value: float) -> None:
        self._nudging = float(value)

    @property
    def force(self) -> torch.Tensor | None:
        return self._force

    def set_force(self, force: torch.Tensor) -> None:
        target = force.detach()
        if tuple(target.shape) != tuple(self._layer.state.shape):
            raise ValueError(
                f"Expected nudging force with shape {tuple(self._layer.state.shape)}, "
                f"got {tuple(target.shape)}."
            )
        self._force = target.to(device=self._layer.state.device, dtype=self._layer.state.dtype)

    def clear(self) -> None:
        self._force = None

    def prepare(
        self,
        *,
        output_gradient: torch.Tensor | None = None,
        loss: torch.Tensor | None = None,
        nudging: float | None = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        if nudging is not None:
            self.nudging = nudging
        if output_gradient is None:
            if loss is None:
                raise ValueError("Nudging.prepare requires either output_gradient or loss.")
            output_gradient = torch.autograd.grad(
                loss,
                self._layer.state,
                retain_graph=retain_graph,
                create_graph=False,
            )[0]
        self.set_force(-output_gradient)
        return self._force

    def eval(self):
        if self._force is None:
            raise RuntimeError("Nudging force has not been prepared.")
        return (
            -self._current_scale
            * self._nudging
            * self._layer.state.mul(self._force).flatten(start_dim=1).sum(dim=1)
        )

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        if self._force is None:
            raise RuntimeError("Nudging force has not been prepared.")
        return -self._current_scale * self._nudging * self._force


class AugmentedFunction(SumSeparableFunction):
    """Augmented block energy with current-force nudging enabled by default."""

    def __init__(
        self,
        energy_fn: SumSeparableFunction,
        nudging_fn: Nudging | None = None,
        *,
        nudging_mode: str = "current",
        current_scale="auto",
    ) -> None:
        interactions = list(getattr(energy_fn, "_interactions", []))
        if not interactions:
            raise TypeError(
                "AugmentedFunction expects an energy function exposing a non-empty '_interactions' list."
            )

        output_layer_getter = getattr(energy_fn, "output_layer", None)
        if nudging_fn is None:
            if output_layer_getter is None:
                raise TypeError("Energy function must expose output_layer() to build default nudging.")
            output_layer = output_layer_getter() if callable(output_layer_getter) else output_layer_getter
            resolved_current_scale = _resolve_current_scale(
                energy_fn,
                output_layer,
                nudging_mode,
                current_scale,
            )
            nudging_fn = Nudging(output_layer, mode=nudging_mode, current_scale=resolved_current_scale)
        else:
            resolved_current_scale = float(getattr(nudging_fn, "current_scale", 1.0))

        self._energy_fn = energy_fn
        self._nudging_fn = nudging_fn
        self._nudging_mode = nudging_fn.mode
        self._current_scale = resolved_current_scale
        self._amplified_current_correction_enabled = (
            self._nudging_mode == "current" and abs(self._current_scale - 1.0) > 1.0e-12
        )
        self._voltage_amp = getattr(energy_fn, "_voltage_amp", None)
        self._current_amp = getattr(energy_fn, "_current_amp", None)

        super().__init__(energy_fn.layers(), energy_fn.params(), interactions + [nudging_fn])

    @property
    def nudging(self) -> Nudging:
        return self._nudging_fn

    @property
    def nudging_mode(self) -> str:
        return self._nudging_mode

    @property
    def current_scale(self) -> float:
        return self._current_scale

    @property
    def amplified_current_correction_enabled(self) -> bool:
        return self._amplified_current_correction_enabled

    def prepare_nudging(self, **kwargs: Any) -> torch.Tensor:
        return self._nudging_fn.prepare(**kwargs)

    def clear_nudging(self) -> None:
        self._nudging_fn.clear()

    def set_device(self, device):
        self._energy_fn.set_device(device)
        self._device = torch.device(device)
        if self._nudging_fn.force is not None:
            self._nudging_fn.set_force(self._nudging_fn.force.to(self._device))
