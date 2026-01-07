import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("softplus")
class Softplus(ActivationBase):
    """Softplus activation wrapper."""
    def __init__(self, beta: int = 1, threshold: int = 20, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
