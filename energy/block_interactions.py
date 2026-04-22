from ..core.interaction import LFunction


class FFCurrentInteraction(LFunction):
    """Linear forcing term that injects a feedforward current into one layer."""

    def __init__(self, layer):
        self._layer = layer
        self._current = None
        super().__init__([layer], [])

    @property
    def current(self):
        return self._current

    def set_current(self, current):
        self._current = current

    def eval(self):
        if self._current is None:
            raise RuntimeError("FFCurrentInteraction current has not been set.")
        return -self._layer.state.mul(self._current).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        if self._current is None:
            raise RuntimeError("FFCurrentInteraction current has not been set.")
        return -self._current
