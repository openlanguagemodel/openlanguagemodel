import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase

@ACTIVATIONS.register("reglu")
class ReGLU(ActivationBase):
    """
    ReGLU activation function.

    Implements the ReGLU variant from "GLU Variants Improve Transformer".
    ReGLU(x, W, V) = ReLU(xW) * (xV)
    Here: ReGLU(x) = ReLU(gate) * value

    Args:
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ReGLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with half the last dimension.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.relu(gate)
