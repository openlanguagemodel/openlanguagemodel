import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("geglu")
class GeGLU(ActivationBase):
	"""Applies GELU activation to the gate before modulating the value stream."""

	def forward(self, x):
		value, gate = x.chunk(2, dim=-1)
		return value * F.gelu(gate)
