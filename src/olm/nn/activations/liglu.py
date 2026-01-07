import torch
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase

@ACTIVATIONS.register("liglu")
class LiGLU(ActivationBase):
    """
    LiGLU activation function.

    Implements the LiGLU variant (Linear GLU).
    LiGLU(x, W, V) = (xW) * (xV)
    Here: LiGLU(x) = gate * value (No activation on gate)

    Args:
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LiGLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with half the last dimension.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * gate
