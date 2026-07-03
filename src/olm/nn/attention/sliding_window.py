import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.embeddings.positional.rope import RotaryPositionalEmbedding
from olm.nn.norms import RMSNorm


class SlidingWindowAttention(AttentionwithRoPEBase):
    """
    Grouped-Query Attention with a fixed sliding window.

    Each query attends only to the ``window_size`` nearest preceding keys
    (plus itself). This reduces memory and compute for long sequences while
    preserving local context.  Interleaved with full-attention layers in a
    3:1 ratio in Step 3.5 Flash, Gemma 3/4, OLMo 3, Tiny Aya, etc.

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        max_seq_len: Maximum sequence length.
        window_size: Sliding window size (tokens). Defaults to 512.
        head_dim: Per-head dimension; inferred from embed_dim / num_heads
            if not provided.
        dropout: Attention dropout probability.
        rope_theta: RoPE base frequency.
        use_qk_norm: Apply RMSNorm to Q and K per head.
        rms_norm_eps: Epsilon for QK-Norm.
        use_bias: Bias on output projection.
        qkv_bias: Bias on QKV projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        window_size: int = 512,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        use_bias: bool = False,
        qkv_bias: bool = False,
    ):
        nn.Module.__init__(self)
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        self.q_dim = num_heads * self.head_dim
        self.kv_dim = num_kv_heads * self.head_dim

        self.q_proj = Linear(embed_dim, self.q_dim, bias=qkv_bias)
        self.k_proj = Linear(embed_dim, self.kv_dim, bias=qkv_bias)
        self.v_proj = Linear(embed_dim, self.kv_dim, bias=qkv_bias)
        self.out_proj = Linear(self.q_dim, embed_dim, bias=use_bias)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, base=rope_theta)

    def _build_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a causal sliding-window boolean mask ``[seq_len, seq_len]``."""
        row = torch.arange(seq_len, device=device)
        col = torch.arange(seq_len, device=device)
        causal = row.unsqueeze(1) >= col.unsqueeze(0)
        window = (row.unsqueeze(1) - col.unsqueeze(0)) < self.window_size
        return causal & window

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = q.shape[2]
        sw_mask = self._build_sliding_window_mask(N, q.device)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(~sw_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout_p, training=self.training)

        return torch.matmul(attn_probs, v)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the attention scores and output.

        Args:
            q (torch.Tensor): Query tensor [batch, heads, seq, head_dim].
            k (torch.Tensor): Key tensor [batch, heads, seq, head_dim].
            v (torch.Tensor): Value tensor [batch, heads, seq, head_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: The attention output [batch, heads, seq, head_dim].
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_groups > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_groups, N, self.head_dim)
            k = k.reshape(B, self.num_heads, N, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_groups, N, self.head_dim)
            v = v.reshape(B, self.num_heads, N, self.head_dim)

        out = self.compute_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, self.q_dim)
        return self.out_proj(out)
