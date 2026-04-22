from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .variable import Variable


class Layer(Variable, ABC):
    _counter = 0

    def __init__(self, shape, batch_size=1, device=None):
        super().__init__(shape)
        self._name = f"Layer_{Layer._counter}"
        self.init_state(batch_size, device)
        Layer._counter += 1

    @property
    def name(self):
        return self._name

    def init_state(self, batch_size, device):
        self._state = torch.zeros((batch_size,) + self._shape, requires_grad=False, device=device)

    @abstractmethod
    def activate(self):
        raise NotImplementedError


class InputLayer(Layer, ABC):
    def activate(self):
        return self._state

    def set_input(self, x):
        self._state = x


class LinearLayer(Layer):
    def activate(self):
        return self._state
