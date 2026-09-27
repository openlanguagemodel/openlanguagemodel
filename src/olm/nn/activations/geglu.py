import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("geglu")
class GeGLU(ActivationBase):
    """
    GeGLU activation function.

    Implements the GeGLU variant from "GLU Variants Improve Transformer".
    GeGLU(x, W, V) = GELU(xW) * (xV)
    Here: GeGLU(x) = GELU(gate) * value

    Args:
        approximate (str, optional): GELU approximation used on the gate, as in
            ``torch.nn.functional.gelu``: ``"none"`` for the exact erf form or
            ``"tanh"`` for the tanh approximation (Hugging Face's
            ``gelu_pytorch_tanh``, used by Gemma). Defaults to ``"none"``.
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def __init__(self, approximate: str = "none", *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"approximate must be 'none' or 'tanh', got {approximate!r}"
            )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GeGLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with half the last dimension.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate, approximate=self.approximate)
