from typing import Optional

import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("gelu")
class GELU(ActivationBase):
    """Gaussian Error Linear Unit wrapper."""

    def __init__(self, approximate: Optional[str] = "tanh", *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, approximate=self.approximate)
