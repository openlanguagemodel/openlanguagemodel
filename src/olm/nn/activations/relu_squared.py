import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("relu2")
class ReLUSquared(ActivationBase):
    """Squared ReLU activation (``relu(x) ** 2``), used by the Nemotron family."""

    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()
