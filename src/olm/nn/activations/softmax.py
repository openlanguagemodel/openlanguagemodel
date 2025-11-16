import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("softmax")
class Softmax(ActivationBase):
	"""Softmax activation over the provided dimension."""

	def __init__(self, dim: int = -1, *, device=None, dtype=None) -> None:
		super().__init__(device=device, dtype=dtype)
		self.dim = dim

	def forward(self, x):
		return F.softmax(x, dim=self.dim)
