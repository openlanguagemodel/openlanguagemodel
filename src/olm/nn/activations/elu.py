import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("elu")
class ELU(ActivationBase):
	"""Exponential Linear Unit with configurable ``alpha``."""

	def __init__(self, alpha: float = 1.0, *, device=None, dtype=None) -> None:
		super().__init__(device=device, dtype=dtype)
		self.alpha = alpha

	def forward(self, x):
		return F.elu(x, alpha=self.alpha)
