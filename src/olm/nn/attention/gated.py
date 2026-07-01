import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.norms import RMSNorm
from olm.nn.embeddings.positional.rope import (
    RotaryPositionalEmbedding,
    PartialRotaryPositionalEmbedding,
)


class GatedAttention(AttentionwithRoPEBase):
    """
    Full softmax attention with a learnable output gate.

    Used by Qwen3.5 as the full-attention layer interleaved with Gated
    DeltaNet (3:1 pattern).  Each head's output is multiplied by
    ``sigmoid(w_gate^T x)`` before the output projection.

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        partial_rotary_factor: Fraction of head_dim that gets RoPE.
            1.0 = full RoPE, <1.0 = partial RoPE (e.g. 0.25 for Qwen3.5).
        use_qk_norm: Apply RMSNorm to Q and K.
        rms_norm_eps: Epsilon for QK-Norm.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 128,
        max_seq_len: int = 262144,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 1.0,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.q_proj = Linear(embed_dim, q_dim, bias=False)
        self.k_proj = Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = Linear(embed_dim, kv_dim, bias=False)
        self.out_proj = Linear(q_dim, embed_dim, bias=False)

        self.output_gate = Linear(embed_dim, q_dim, bias=False)

        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        if partial_rotary_factor < 1.0:
            self.rope = PartialRotaryPositionalEmbedding(
                head_dim,
                rotary_percentage=partial_rotary_factor,
                base=rope_theta,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len, base=rope_theta)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_groups > 1:
            k = k[:, :, None].expand(B, self.num_kv_heads, self.num_groups, N, self.head_dim)
            k = k.reshape(B, self.num_heads, N, self.head_dim)
            v = v[:, :, None].expand(B, self.num_kv_heads, self.num_groups, N, self.head_dim)
            v = v.reshape(B, self.num_heads, N, self.head_dim)

        attn_out = self.compute_attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)

        gate = torch.sigmoid(self.output_gate(x))
        attn_out = attn_out * gate

        return self.out_proj(attn_out)
