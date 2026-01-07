from torch import nn
import torch 
from typing import Optional
from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase

@NORMS.register("rms_norm")
class RMSNorm(NormBase):
    """
    RMSNorm (Root Mean Square Layer Normalization) layer.

    Implements RMSNorm as described in "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
    A simplified version of LayerNorm that scales invariance properties.

    Args:
        d_model (int): The dimension of the model to normalize.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__(d_model, device=device, dtype=dtype)
        self.eps = eps
        self.weight = nn.Parameter(torch.full((d_model,), 1., device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Normalized output tensor of the same shape.
        """
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS_a = sqrt( (1/d_model) * sum_{i=1}^{d_model} x_i^2 + eps )
        RMS_a = torch.sqrt( ( torch.sum(x**2, dim=2) / self.d_model) + self.eps)
        result = ( x  / RMS_a.unsqueeze(-1) ) * self.weight
        return result.to(in_dtype)
