import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("mish")
class Mish(ActivationBase):
    """Mish activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Mish(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
