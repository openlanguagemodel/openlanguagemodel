import torch.nn.functional as F

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("reglu")
class ReGLU(ActivationBase):
	"""Applies ReLU to the gate before modulation."""

	def forward(self, x):
		value, gate = x.chunk(2, dim=-1)
		return value * F.relu(gate)
