from __future__ import annotations

from abc import ABC, abstractmethod
import copy

import torch


class Function(ABC):
    def __init__(self, layers, params):
        self._layers = list(layers)
        self._params = list(params)
        self._device = None

    def params(self):
        return self._params

    def layers(self):
        return self._layers

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    def grad_layer_fn(self, layer):
        return lambda: self._grad(layer, mean=False)

    def grad_param_fn(self, param):
        return lambda: self._grad(param, mean=True)

    def second_fn(self, param):
        return lambda direction: self._second_derivative(param, direction)

    def _grad(self, variable, mean=False):
        requires_grad = variable.state.requires_grad
        if not requires_grad:
            variable.state.requires_grad_(True)
        value = torch.mean(self.eval()) if mean else torch.sum(self.eval())
        grad = torch.autograd.grad(value, variable.state, create_graph=requires_grad)[0]
        if not requires_grad:
            variable.state.requires_grad_(False)
        return grad

    def _second_derivative(self, param, direction):
        for layer in self._layers:
            layer.state.requires_grad_(True)
        for other_param in self._params:
            other_param.state.requires_grad_(True)
        value = torch.mean(self.eval())
        layer_grads = torch.autograd.grad(value, [layer.state for layer in self._layers], create_graph=True)
        direction_list = [direction[layer.name] for layer in self._layers]
        param_grad = torch.autograd.grad(layer_grads, param.state, grad_outputs=direction_list)[0]
        for layer in self._layers:
            layer.state.requires_grad_(False)
        for other_param in self._params:
            other_param.state.requires_grad_(False)
        return param_grad

    def set_device(self, device):
        self._device = device
        for layer in self._layers:
            layer.set_device(device)
        for param in self._params:
            param.set_device(device)

    def to(self, device):
        fn = copy.deepcopy(self)
        fn.set_device(device)
        return fn

    def save(self, path):
        torch.save([param.state for param in self._params], path)

    def load(self, path):
        params = torch.load(path, map_location=torch.device(self._device))
        for param, state in zip(self._params, params):
            param.state = state


class QFunction(Function, ABC):
    @abstractmethod
    def a_coef_fn(self, layer):
        raise NotImplementedError

    @abstractmethod
    def b_coef_fn(self, layer):
        raise NotImplementedError


class LFunction(QFunction, ABC):
    def a_coef_fn(self, layer):
        return None

    def b_coef_fn(self, layer):
        return self.grad_layer_fn(layer)


class BiasInteraction(LFunction):
    def __init__(self, layer, bias):
        self._layer = layer
        self._bias = bias
        super().__init__([layer], [bias])

    def eval(self):
        return -self._layer.state.mul(self._bias.get()).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._b_coef_layer

    def grad_param_fn(self, param):
        if param is not self._bias:
            raise ValueError("Requested gradient for an unmanaged parameter.")
        return self._grad_bias

    def _b_coef_layer(self):
        return -self._bias.get()

    def _grad_bias(self):
        return -self._layer.state.mean(dim=0)


class HardSigmoidNonLinearInteraction(Function):
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        layer_index = int(layer._name[-1])
        scale = (current_amp / voltage_amp) ** (layer_index - 1)
        self._g_on = params.get("g_on") * scale
        self._g_off = params.get("g_off") * scale
        self.v_min = params.get("v_min")
        self.v_max = params.get("v_max")
        super().__init__([layer], [])

    def eval(self):
        v = self._layer.state
        off_mask = (v >= self.v_min) & (v <= self.v_max)
        pos_on_mask = v > self.v_max
        neg_on_mask = v < self.v_min

        energy_off = 0.5 * self._g_off * (v**2) * off_mask
        energy_on = torch.zeros_like(v)
        energy_on[pos_on_mask] = 0.5 * self._g_on * (v[pos_on_mask] - self.v_max) ** 2
        energy_on[neg_on_mask] = 0.5 * self._g_on * (v[neg_on_mask] - self.v_min) ** 2
        return (energy_off + energy_on).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        grad = torch.zeros_like(v)
        off_mask = (v >= self.v_min) & (v <= self.v_max)
        pos_on_mask = v > self.v_max
        neg_on_mask = v < self.v_min
        grad[off_mask] = self._g_off * v[off_mask]
        grad[pos_on_mask] = self._g_on * (v[pos_on_mask] - self.v_max)
        grad[neg_on_mask] = self._g_on * (v[neg_on_mask] - self.v_min)
        return grad

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class LpwNonLinearInteraction(Function):
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        layer_index = int(layer._name[-1])
        scale = (current_amp / voltage_amp) ** (layer_index - 1)
        self._g = params.get("diode_conductance") * scale
        self._v_off = params.get("v_off", 0.0)
        super().__init__([layer], [])

    def _split(self, v):
        dim = v.shape[1] // 2
        return v[:, :dim], v[:, dim:]

    def eval(self):
        v = self._layer.state
        exc, inh = self._split(v)
        exc_excess = torch.clamp(exc - self._v_off, min=0.0)
        inh_excess = torch.clamp(-inh - self._v_off, min=0.0)
        energy = 0.5 * self._g * (exc_excess**2)
        energy = energy + 0.5 * self._g * (inh_excess**2)
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        exc, inh = self._split(v)
        exc_excess = torch.clamp(exc - self._v_off, min=0.0)
        inh_excess = torch.clamp(-inh - self._v_off, min=0.0)
        return torch.cat((self._g * exc_excess, -self._g * inh_excess), dim=1)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class DoubleQuadraticNonLinearInteraction(Function):
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        layer_index = int(layer._name[-1])
        self.c = params.get("diode_conductance", params.get("c")) * (current_amp / voltage_amp) ** (layer_index - 1)
        self.v_off = params.get("v_off", 0.0)
        super().__init__([layer], [])

    def eval(self):
        v = self._layer.state
        excess = torch.clamp(torch.abs(v) - self.v_off, min=0.0)
        energy = (self.c / 3.0) * excess**3
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        excess = torch.clamp(torch.abs(v) - self.v_off, min=0.0)
        return self.c * excess**2 * torch.sign(v)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class DoubleExponentialNonLinearInteraction(Function):
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        layer_index = int(layer._name[-1])
        scaling_factor = (current_amp / voltage_amp) ** (layer_index - 1)
        self.I_s = params.get("I_s") * scaling_factor
        self.vt = params.get("V_t")
        self.v_off = params.get("V_off")
        super().__init__([layer], [])

    def eval(self):
        v = self._layer.state
        abs_v = torch.abs(v)
        Is = torch.as_tensor(self.I_s, dtype=v.dtype, device=v.device)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)
        excess = torch.clamp(abs_v - voff, min=0.0)
        energy = Is * vt * (torch.exp(excess / vt) - torch.exp(-voff / vt)) - Is * excess
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        abs_v = torch.abs(v)
        grad_abs = self.I_s * (torch.exp((abs_v - self.v_off) / self.vt) - 1.0)
        return grad_abs * torch.sign(v)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class SingleExponentialNonLinearInteraction(Function):
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        layer_index = int(layer._name[-1])
        scaling_factor = (current_amp / voltage_amp) ** (layer_index - 1)
        self.I_s = params.get("I_s") * scaling_factor
        self.vt = params.get("V_t")
        self.v_off = params.get("V_off")
        super().__init__([layer], [])

    def _split_mask(self, v_flat):
        features = v_flat.shape[1]
        reverse_mask_nodes = torch.arange(features, device=v_flat.device) >= (features // 2)
        return reverse_mask_nodes.unsqueeze(0).expand_as(v_flat)

    def eval(self):
        v = self._layer.state
        v_flat = v.reshape(v.shape[0], -1)
        Is = torch.as_tensor(self.I_s, dtype=v.dtype, device=v.device)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)
        excess_fwd = torch.clamp(v_flat - voff, min=0.0)
        excess_rev = torch.clamp(-v_flat - voff, min=0.0)
        energy_fwd = Is * vt * (torch.exp(excess_fwd / vt) - torch.exp(-voff / vt)) - Is * excess_fwd
        energy_rev = Is * vt * (torch.exp(excess_rev / vt) - torch.exp(-voff / vt)) - Is * excess_rev
        return torch.where(self._split_mask(v_flat), energy_rev, energy_fwd).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        v_flat = v.reshape(v.shape[0], -1)
        Is = torch.as_tensor(self.I_s, dtype=v.dtype, device=v.device)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)
        excess_fwd = torch.clamp(v_flat - voff, min=0.0)
        excess_rev = torch.clamp(-v_flat - voff, min=0.0)
        grad_fwd = Is * (torch.exp(excess_fwd / vt) - 1.0)
        grad_rev = -Is * (torch.exp(excess_rev / vt) - 1.0)
        return torch.where(self._split_mask(v_flat), grad_rev, grad_fwd).reshape_as(v)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class SumSeparableFunction(Function):
    def __init__(self, layers, params, interactions):
        super().__init__(layers, params)
        self._interactions = list(interactions)

    def eval(self):
        return sum(interaction.eval() for interaction in self._interactions)

    def grad_layer_fn(self, layer):
        fns = [interaction.grad_layer_fn(layer) for interaction in self._interactions if layer in interaction.layers()]
        return lambda: sum(fn() for fn in fns)

    def grad_param_fn(self, param):
        fns = [interaction.grad_param_fn(param) for interaction in self._interactions if param in interaction.params()]
        return lambda: sum(fn() for fn in fns)

    def second_fn(self, param):
        fns = [interaction.second_fn(param) for interaction in self._interactions if param in interaction.params()]
        return lambda direction: sum(fn(direction) for fn in fns)

    def a_coef_fn(self, layer):
        fns = [interaction.a_coef_fn(layer) for interaction in self._interactions if layer in interaction.layers()]
        fns = [fn for fn in fns if fn is not None]
        return lambda: sum(fn() for fn in fns)

    def b_coef_fn(self, layer):
        fns = [interaction.b_coef_fn(layer) for interaction in self._interactions if layer in interaction.layers()]
        return lambda: sum(fn() for fn in fns)
