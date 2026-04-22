from __future__ import annotations

from abc import ABC
import math

import torch

from .variable import Variable


class Parameter(Variable, ABC):
    def __init__(self, shape, device, non_negative=True, min_cond=None, max_cond=None):
        super().__init__(shape)
        self._state = torch.empty(*shape, dtype=torch.float32, device=device)
        self._non_negative = non_negative
        self.min_cond = min_cond
        self.max_cond = max_cond

    def get(self):
        return self._state

    def clamp_(self):
        clamp_min = self.min_cond
        clamp_max = self.max_cond
        if self._non_negative:
            if clamp_min is None:
                clamp_min = 0.0
            if clamp_max is None:
                clamp_max = float("inf")
        if clamp_min is not None or clamp_max is not None:
            self._state.clamp_(min=clamp_min, max=clamp_max)


class Bias(Parameter):
    _counter = 0

    def __init__(self, shape, gain, device):
        super().__init__(shape, device=device)
        self.init_state(gain)
        self.name = f"Bias_{Bias._counter}"
        Bias._counter += 1

    def init_state(self, gain):
        torch.nn.init.uniform_(self._state, -gain, +gain)


class DenseWeight(Parameter):
    _counter = 0

    def __init__(
        self,
        layer_pre_shape,
        layer_post_shape,
        gain,
        device,
        clamp=False,
        clamp_min=None,
        clamp_max=None,
        init_mode="kaiming_uniform",
    ):
        shape = tuple(layer_pre_shape) + tuple(layer_post_shape)
        super().__init__(shape, device=device, non_negative=clamp, min_cond=clamp_min, max_cond=clamp_max)
        self._layer_pre_shape = tuple(layer_pre_shape)
        self._layer_post_shape = tuple(layer_post_shape)
        self.init_state(gain, mode=init_mode)
        self.clamp_()
        self.name = f"DenseWeight_{DenseWeight._counter}"
        DenseWeight._counter += 1

    def init_state(self, gain, mode="kaiming_uniform"):
        size_pre = math.prod(self._layer_pre_shape)
        size_post = math.prod(self._layer_post_shape)

        if mode == "xavier_uniform":
            scale = gain * 0.5 * math.sqrt(6.0 / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == "xavier_normal":
            scale = gain * 0.5 * math.sqrt(2.0 / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == "kaiming_uniform":
            scale = gain * math.sqrt(1.0 / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == "bounded_uniform":
            upper = self.max_cond if self.max_cond is not None else gain
            torch.nn.init.uniform_(self._state, 0.0, upper)
        elif mode == "kendall":
            lower = 1e-7
            upper = 0.08 / math.sqrt(size_pre + size_post)
            torch.nn.init.uniform_(self._state, lower, upper)
        else:
            scale = gain * 0.5 * math.sqrt(1.0 / size_pre)
            torch.nn.init.normal_(self._state, std=scale)


class ConvWeight(Parameter):
    _counter = 0

    def __init__(
        self,
        shape,
        gain,
        device,
        clamp=False,
        clamp_min=None,
        clamp_max=None,
        init_mode="kaiming_uniform",
    ):
        shape = tuple(shape)
        if len(shape) != 4:
            raise ValueError(f"ConvWeight expects shape (out_channels, in_channels, kh, kw), got {shape}.")
        super().__init__(shape, device=device, non_negative=clamp, min_cond=clamp_min, max_cond=clamp_max)
        self.init_state(gain, init_mode=init_mode)
        self.clamp_()
        self.name = f"ConvWeight_{ConvWeight._counter}"
        ConvWeight._counter += 1

    def init_state(self, gain, init_mode="kaiming_uniform"):
        channels_out, channels_in, kh, kw = self._shape
        size_pre = channels_in * kh * kw
        size_post = channels_out

        if init_mode == "xavier_uniform":
            scale = gain * 0.5 * math.sqrt(6.0 / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif init_mode == "xavier_normal":
            scale = gain * 0.5 * math.sqrt(2.0 / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif init_mode == "kaiming_uniform":
            scale = gain * math.sqrt(1.0 / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:
            scale = gain * 0.5 * math.sqrt(1.0 / size_pre)
            torch.nn.init.normal_(self._state, std=scale)
