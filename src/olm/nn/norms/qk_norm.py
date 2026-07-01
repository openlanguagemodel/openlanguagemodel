import torch
import torch.nn as nn
from typing import Optional

from olm.nn.norms.base import NormBase
from olm.nn.norms.rms_norm import RMSNorm


class QKNorm(NormBase):
    """
    Query-Key Normalization applied independently per attention head.

    Normalizes Q and K projections using RMSNorm before computing attention
    scores. This stabilizes training at large scale by preventing attention
    logit growth, as used in Qwen2/3, Gemma 2, MiniMax M2.5, Sarvam, and
    many recent MoE architectures.

    Reference: "Scaling Vision Transformers" (https://arxiv.org/abs/2106.04560)

    Args:
        head_dim (int): Dimension of each attention head (used as d_model).
        eps (float): Small constant for numerical stability in RMSNorm.
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(head_dim, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=eps, device=device, dtype=dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize Q and K tensors per head.

        Args:
            q: Query tensor of shape ``[batch, seq_len, num_heads, head_dim]``
               or ``[batch, num_heads, seq_len, head_dim]``.
            k: Key tensor of shape ``[batch, seq_len, num_kv_heads, head_dim]``
               or ``[batch, num_kv_heads, seq_len, head_dim]``.

        Returns:
            Tuple of normalized (q, k) with the same shapes as input.
        """
        return self.q_norm(q), self.k_norm(k)
