from torch import nn
import torch 
from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase

@NORMS.register("rms_norm")
class RMSNorm(NormBase):
    """RMSNorm layer as described in https://arxiv.org/abs/1910.07467"""
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__(d_model, device=device, dtype=dtype)
        self.eps = eps
        self.weight = nn.Parameter(torch.full((d_model,), 1, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS_a = sqrt( (1/d_model) * sum_{i=1}^{d_model} x_i^2 + eps )
        RMS_a = torch.sqrt( ( torch.sum(x**2, dim=2) / self.d_model) + self.eps)
        result = ( x  / RMS_a.unsqueeze(-1) ) * self.weight
        return result.to(in_dtype)
