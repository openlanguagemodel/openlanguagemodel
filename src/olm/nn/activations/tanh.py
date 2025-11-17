import torch

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("tanh")
class Tanh(ActivationBase):
	"""Hyperbolic tangent activation."""

	def forward(self, x):
		return torch.tanh(x)
