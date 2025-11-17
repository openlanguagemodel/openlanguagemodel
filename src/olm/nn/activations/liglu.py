from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("liglu")
class LiGLU(ActivationBase):
	"""Linear Gated Linear Unit keeps the gate linear (identity)."""

	def forward(self, x):
		value, gate = x.chunk(2, dim=-1)
		return value * gate
