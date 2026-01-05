import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("tanh")
class Tanh(ActivationBase):
    """Tanh activation wrapper."""
    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
