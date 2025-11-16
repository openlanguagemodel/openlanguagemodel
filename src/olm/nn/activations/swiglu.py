import torch

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase

import torch.nn.functional as F


@ACTIVATIONS.register("swiglu")
class SwiGLU(ActivationBase):
    """SwiGLU activation: (x1 * SiLU(x2)) where x is split evenly along last dim."""

    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)