import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("glu")
class GLU(ActivationBase):
	"""Wrapper for ``torch.nn.functional.glu``."""

	def __init__(self, dim: int = -1, *, device=None, dtype=None) -> None:
		super().__init__(device=device, dtype=dtype)
		self.dim = dim

	def forward(self, x):
		return F.glu(x, dim=self.dim)
