from __future__ import annotations

from abc import ABC, abstractmethod
import copy


class Variable(ABC):
    def __init__(self, shape):
        self._shape = tuple(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def set_device(self, device):
        self._state = self._state.to(device)

    def to(self, device):
        variable = copy.deepcopy(self)
        variable._state = variable._state.to(device)
        return variable

    @abstractmethod
    def init_state(self, *args, **kwargs):
        raise NotImplementedError
