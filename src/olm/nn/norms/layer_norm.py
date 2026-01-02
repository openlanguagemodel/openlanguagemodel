from torch import nn
import torch 
from typing import Optional
from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase

@NORMS.register("layer_norm")
class LayerNorm(NormBase):
    """
    Layer Normalization layer.

    Implements Layer Normalization as described in "Layer Normalization" (https://arxiv.org/abs/1607.06450).
    Normalizes the input across the features dimension.

    Args:
        d_model (int): The dimension of the model to normalize.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.

    Attributes:
        gamma (nn.Parameter): Learnable scale parameter.
        beta (nn.Parameter): Learnable shift parameter.
    """
    def __init__(self, d_model: int, eps: float=1e-5, elementwise_affine: bool = True, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__(d_model, device=device, dtype=dtype)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.full((d_model,), 1., device=device, dtype=dtype))
            self.beta = nn.Parameter(torch.full((d_model,), 0., device=device, dtype=dtype))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LayerNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Normalized output tensor of the same shape.
        """
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=2, keepdim=True)  # (batch_size, sequence_length, 1)
        variance = x.var(dim=2, keepdim=True, unbiased=False)  # (batch_size, sequence_length, 1)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)  # (batch_size, sequence_length, d_model)
        
        if self.elementwise_affine:
            result = x_normalized * self.gamma + self.beta
        else:
            result = x_normalized
            
        return result.to(in_dtype)
