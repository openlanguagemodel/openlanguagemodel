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
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GeGLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with half the last dimension.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)
