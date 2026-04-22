import torch

from digital_drn.core.interaction import HardSigmoidNonLinearInteraction
from digital_drn.core.minimizer import HardSigmoidUpdater
from digital_drn.core.resistive import NonlinearResistiveLayer


class _StaticQuadratic:
    def __init__(self, layer, a: torch.Tensor, b: torch.Tensor):
        self._layer = layer
        self._a = a
        self._b = b

    def a_coef_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Unknown layer.")
        return lambda: self._a

    def b_coef_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Unknown layer.")
        return lambda: self._b

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Unknown layer.")
        return lambda: torch.zeros_like(self._layer.state)


def _stationarity_residual(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    params: dict[str, float],
) -> torch.Tensor:
    layer = NonlinearResistiveLayer((a.shape[1],), batch_size=a.shape[0], non_linearity="hard_sigmoid")
    fn = _StaticQuadratic(layer, a, b)
    updater = HardSigmoidUpdater(layer, fn, params)
    v = updater.pre_activate()
    layer.state = v.detach().clone()
    interaction = HardSigmoidNonLinearInteraction(layer, params, voltage_amp=1.0, current_amp=1.0)
    grad_nl = interaction.grad_layer_fn(layer)()
    return 2.0 * a * v + b + grad_nl


def test_hard_sigmoid_updater_matches_off_region_interaction_gradient():
    a = torch.full((1, 4), 1.0, dtype=torch.float32)
    b = torch.tensor([[-0.4, 0.2, -0.6, 0.8]], dtype=torch.float32)
    params = {"g_on": 0.5, "g_off": 0.1, "v_min": -0.5, "v_max": 0.5}

    residual = _stationarity_residual(a=a, b=b, params=params)

    assert residual.abs().max().item() < 1.0e-6


def test_hard_sigmoid_updater_matches_on_region_interaction_gradient():
    a = torch.full((1, 4), 1.0, dtype=torch.float32)
    b = torch.tensor([[-2.0, 2.0, -3.0, 3.0]], dtype=torch.float32)
    params = {"g_on": 0.5, "g_off": 0.1, "v_min": -0.5, "v_max": 0.5}

    residual = _stationarity_residual(a=a, b=b, params=params)

    assert residual.abs().max().item() < 1.0e-6
