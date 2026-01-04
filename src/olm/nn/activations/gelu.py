import torch
import torch.nn.functional as F
from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("gelu")
class GELU(ActivationBase):
    """
    GELU (Gaussian Error Linear Unit) activation function.

    Applies the GELU activation function element-wise. GELU is a smooth approximation
    to the ReLU activation and is commonly used in modern transformer models like BERT and GPT.

    Equation:
        GELU(x) = x * Φ(x)
        where Φ(x) is the cumulative distribution function of the standard normal distribution.

    Args:
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GELU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with GELU applied.
        """
        return F.gelu(x)
