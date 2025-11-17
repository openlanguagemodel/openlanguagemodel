import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("selu")
class SELU(ActivationBase):
    """Scaled Exponential Linear Unit."""

    def forward(self, x):
        return F.selu(x)
