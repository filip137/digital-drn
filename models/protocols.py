from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..core.parameter import Parameter


@runtime_checkable
class ResistiveTrainableModel(Protocol):
    """Protocol for models trainable by the digital_drn BP trainer.

    Implementations are expected to be ``nn.Module``-like and to expose the
    resistive-parameter lifecycle used by the shared trainer. This keeps
    transformer-style models from needing to inherit the sequential
    ``DigitalDRNNet`` container just to participate in BP training.
    """

    training: bool

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def train(self, mode: bool = True): ...

    def eval(self): ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]: ...

    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Module]]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True): ...

    def set_device(self, device: torch.device | str): ...

    def enable_resistive_grad_(self, enabled: bool = True): ...

    def zero_resistive_grad_(self, set_to_none: bool = True): ...

    def clamp_resistive_params_(self): ...

    def detach_state_(self): ...

    def resistive_params(self) -> Sequence[Parameter]: ...

    def resistive_param_states(self) -> Sequence[torch.Tensor]: ...

    def named_resistive_parameters(self) -> Iterator[tuple[str, torch.Tensor]]: ...

    def optimizer_tensors(self) -> Sequence[torch.Tensor | nn.Parameter]: ...

    def optimizer_param_groups(self) -> list[dict[str, Any]]: ...
