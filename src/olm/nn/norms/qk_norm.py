import torch
import torch.nn as nn
from typing import Optional

from olm.nn.norms.rms_norm import RMSNorm


class QKNorm(nn.Module):
    """
    Query-Key Normalization applied independently per attention head.

    Normalizes Q and K projections using RMSNorm before computing attention
    scores. This stabilizes training at large scale by preventing attention
    logit growth, as used in Qwen2/3, Gemma 2, MiniMax M2.5, Sarvam, and
    many recent MoE architectures.

    Reference: "Scaling Vision Transformers" (https://arxiv.org/abs/2106.04560)

    Args:
        head_dim (int): Dimension of each attention head.
        eps (float): Small constant for numerical stability in RMSNorm.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

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
