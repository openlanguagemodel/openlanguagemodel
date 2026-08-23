from typing import Optional

import torch
from torch import nn

from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase


@NORMS.register("adaptive_layer_norm")
class AdaptiveLayerNorm(NormBase):
    """
    Adaptive Layer Normalization (AdaLN).

    Computes:

        y = gamma(cond) * LN(x) + beta(cond)

    where gamma and beta are generated from a
    conditioning vector.

    Args:
        d_model (int):
            Hidden dimension.

        cond_dim (int):
            Dimension of conditioning vector.

        eps (float):
            Numerical stability constant.
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.eps = eps

        self.norm = nn.LayerNorm(
            d_model,
            elementwise_affine=False,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.gamma_proj = nn.Linear(
            cond_dim,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.beta_proj = nn.Linear(
            cond_dim,
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Shape:
                (batch_size, seq_len, d_model)

            cond:
                Shape:
                (batch_size, cond_dim)

        Returns:
            Tensor of shape:
            (batch_size, seq_len, d_model)
        """

        x_norm = self.norm(x)

        gamma = self.gamma_proj(cond).unsqueeze(1)
        beta = self.beta_proj(cond).unsqueeze(1)

        return gamma * x_norm + beta
