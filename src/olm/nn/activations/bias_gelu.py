import torch
import torch.nn as nn
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("bias_gelu")
class BiasGELU(ActivationBase):
    """Bias + GELU activation: GELU(x + bias)."""
    def __init__(self, hidden_dim: int, approximate: str = "none", *, device=None, dtype=None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.bias = nn.Parameter(torch.zeros(hidden_dim, **factory_kwargs))
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.bias)
