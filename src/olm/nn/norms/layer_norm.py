from torch import nn
import torch 
from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase

@NORMS.register("layer_norm")
class LayerNorm(NormBase):
    """LayerNorm layer as described in https://arxiv.org/abs/1607.06450"""
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__(d_model, device=device, dtype=dtype)
        self.eps = eps
        self.gamma = nn.Parameter(torch.full((d_model,), 1, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.full((d_model,), 0, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=2, keepdim=True)  # (batch_size, sequence_length, 1)
        variance = x.var(dim=2, keepdim=True, unbiased=False)  # (batch_size, sequence_length, 1)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)  # (batch_size, sequence_length, d_model)
        result = x_normalized * self.gamma + self.beta  # (batch_size, sequence_length, d_model)
        return result.to(in_dtype)
