import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("silu")
class SiLU(ActivationBase):
	"""``SiLU`` (or Swish) activation."""

	def forward(self, x):
		return F.silu(x)