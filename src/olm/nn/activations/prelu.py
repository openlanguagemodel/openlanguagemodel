import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("prelu")
class PReLU(ActivationBase):
    """PReLU activation wrapper."""
    def __init__(self, num_parameters: int = 1, init: float = 0.25, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        # PReLU parameters need to be on the correct device/dtype
        self.act = nn.PReLU(num_parameters=num_parameters, init=init).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
