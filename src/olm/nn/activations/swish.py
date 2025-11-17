import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("swish")
class Swish(ActivationBase):
	"""Alias for ``SiLU`` activation."""

	def forward(self, x):
		return F.silu(x)
