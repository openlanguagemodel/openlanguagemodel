import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("swiglu")
class SwiGLU(ActivationBase):
    """
    SwiGLU activation function.

    Implements the SwiGLU activation as described in "GLU Variants Improve Transformer".
    It applies the SiLU activation to one half of the input (the gate) and multiplies it
    by the other half (the value).

    Equation:
        SwiGLU(x, W, V) = Swish_1(xW) * (xV)
        Here, we assume the input `x` is already projected/concatenated such that we chunk it.
        So: SwiGLU(x) = (x_1 * SiLU(x_2)) where x = [x_1, x_2]

    Args:
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU.

        Args:
            x (torch.Tensor): Input tensor. Expected to have an even last dimension size.

        Returns:
            torch.Tensor: Output tensor with half the last dimension of the input.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)
