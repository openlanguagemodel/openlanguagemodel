import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("relu")
class ReLU(ActivationBase):
    """
    ReLU (Rectified Linear Unit) activation function.

    Applies the ReLU activation function element-wise: ReLU(x) = max(0, x).
    This is a standard activation used in many neural networks, including the original GPT-2.

    Args:
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ReLU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with ReLU applied.
        """
        return F.relu(x)
