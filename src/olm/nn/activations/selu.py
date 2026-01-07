import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("selu")
class SELU(ActivationBase):
    """SELU activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.SELU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
