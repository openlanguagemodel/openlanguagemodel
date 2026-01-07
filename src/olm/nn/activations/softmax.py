from typing import Optional
import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("softmax")
class Softmax(ActivationBase):
    """Softmax activation wrapper."""
    def __init__(self, dim: Optional[int] = None, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
