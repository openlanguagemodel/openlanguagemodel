from typing import Optional
import torch
from torch import nn

from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase


@NORMS.register("weight_norm")
class WeightNorm(NormBase):
    """
    Weight Normalization layer.

    Reparameterizes an input feature vector as:

        y = g * x / ||x||

    where:

        ||x|| = sqrt(sum(x²) + eps)

    Unlike PixelNorm, which normalizes by RMS,
    WeightNorm normalizes by the L2 norm and
    introduces a learnable scaling parameter g.

    Args:
        d_model (int):
            Feature dimension.

        eps (float, optional):
            Numerical stability constant.

        device (torch.device, optional):
            Device placement.

        dtype (torch.dtype, optional):
            Tensor dtype.
    """

    def __init__(self, d_model: int, eps: float = 1e-8, 
                device: Optional[torch.device] = None, 
                dtype: Optional[torch.dtype] = None,):
        
        super().__init__(d_model, device=device,dtype=dtype)

        self.eps = eps

        self.g = nn.Parameter(torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype

        x = x.to(torch.float32)

        norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)

        x = x / norm

        x = x * self.g

        return x.to(in_dtype)

