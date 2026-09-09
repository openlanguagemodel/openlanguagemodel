from torch import nn
import torch
from typing import Optional

from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase


@NORMS.register("pixel_norm")
class PixelNorm(NormBase):
    """
    Pixel Normalization layer.

    Implements Pixel Normalization, which scales each feature vector by its
    root mean square (RMS) value. Unlike LayerNorm, PixelNorm does not
    subtract the mean and does not contain any learnable parameters.

    The normalization is computed as:

        y = x / sqrt(mean(x²) + eps)

    where the mean is calculated across the feature dimension.

    Args:
        d_model (int):
            The dimension of the feature vector to normalize.

        eps (float, optional):
            Small constant added to the denominator for numerical stability.
            Defaults to 1e-8.

        device (torch.device, optional):
            Device on which the module parameters and computations are
            initialized. Defaults to None.

        dtype (torch.dtype, optional):
            Data type used for module initialization. Defaults to None.

    Attributes:
        eps (float):
            Numerical stability constant used during normalization.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(d_model, device=device, dtype=dtype)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PixelNorm.

    Args:
        x (torch.Tensor):
            Input tensor of shape
            (batch_size, sequence_length, d_model).

    Returns:
        torch.Tensor:
            Tensor of the same shape as the input, where each feature
            vector has been normalized by its RMS magnitude.
        """
        in_dtype = x.dtype

        x = x.to(torch.float32)

        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return x.to(in_dtype)