import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.norms import RMSNorm


class LightningAttention(nn.Module):
    """
    Lightning Attention -- a linear attention variant used by Ling 2.5.

    Replaces softmax attention with a linear kernel approximation, yielding
    O(N) complexity for sequence length N.  During training / prefill the
    computation is parallelised; during single-token decode the cost is
    O(1) per step via a fixed-size recurrent state.

    The core idea: instead of ``softmax(QK^T)V``, compute ``(phi(Q) @ S)``
    where ``S = sum_t phi(k_t) @ v_t^T`` is a running outer-product state.

    Ling 2.5 uses 70 Lightning Attention layers interleaved with 10 MLA
    layers (7:1 ratio).

    Reference: "Lightning Attention-2" (arXiv:2401.04658)

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.
        dropout: Output dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads

        total_dim = num_heads * self.head_dim

        self.q_proj = Linear(embed_dim, total_dim, bias=False)
        self.k_proj = Linear(embed_dim, total_dim, bias=False)
        self.v_proj = Linear(embed_dim, total_dim, bias=False)
        self.out_proj = Linear(total_dim, embed_dim, bias=False)

        self.output_gate = Linear(embed_dim, total_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map: phi(x) = elu(x) + 1."""
        return F.elu(x, alpha=1.0) + 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Causal linear attention (parallel mode).

        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self._elu_feature_map(q)
        k = self._elu_feature_map(k)

        q = q.transpose(1, 2)  # [B, H, N, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        outputs = []
        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                         device=x.device, dtype=x.dtype)
        z = torch.zeros(B, self.num_heads, self.head_dim,
                        device=x.device, dtype=x.dtype)

        for t in range(N):
            k_t = k[:, :, t]    # [B, H, d]
            v_t = v[:, :, t]    # [B, H, d]
            q_t = q[:, :, t]    # [B, H, d]

            S = S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            z = z + k_t

            num = torch.einsum("bhd,bhde->bhe", q_t, S)
            denom = torch.einsum("bhd,bhd->bh", q_t, z).unsqueeze(-1).clamp(min=1e-6)
            outputs.append(num / denom)

        out = torch.stack(outputs, dim=2)  # [B, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)

        gate = torch.sigmoid(self.output_gate(x))
        out = out * gate

        return self.dropout(self.out_proj(out))
