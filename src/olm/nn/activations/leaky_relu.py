import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("leaky_relu")
class LeakyReLU(ActivationBase):
    """LeakyReLU activation wrapper."""
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
