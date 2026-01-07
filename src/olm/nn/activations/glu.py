import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("glu")
class GLU(ActivationBase):
    """GLU activation wrapper."""
    def __init__(self, dim: int = -1, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.GLU(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
