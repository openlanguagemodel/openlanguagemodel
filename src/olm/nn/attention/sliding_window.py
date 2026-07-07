import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.embeddings.positional.rope import (
    PartialRotaryPositionalEmbedding,
    RotaryPositionalEmbedding,
)
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
        use_rope: bool = True,
        partial_rotary_factor: float = 1.0,
        use_attention_sink: bool = False,
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
        self.use_rope = use_rope
        self.use_attention_sink = use_attention_sink

        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                "partial_rotary_factor must be in (0.0, 1.0], "
                f"got {partial_rotary_factor}"
            )

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

        if use_rope:
            if partial_rotary_factor < 1.0:
                self.rope = PartialRotaryPositionalEmbedding(
                    self.head_dim,
                    rotary_percentage=partial_rotary_factor,
                    base=rope_theta,
                    max_seq_len=max_seq_len,
                )
            else:
                self.rope = RotaryPositionalEmbedding(
                    self.head_dim, max_seq_len, base=rope_theta
                )
        else:
            self.rope = None

        if use_attention_sink:
            self.attention_sink = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("attention_sink", None)

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
        attn_mask = sw_mask

        if mask is not None:
            user_mask = attention_mask_to_bool(mask, device=q.device)
            attn_mask = user_mask & sw_mask

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        if self.use_attention_sink:
            sink_scores = self.attention_sink.view(1, self.num_heads, 1, 1)
            sink_scores = sink_scores.expand(q.size(0), self.num_heads, N, 1)
            attn_scores = torch.cat([attn_scores, sink_scores], dim=-1)
            sink_values = torch.zeros(
                v.size(0),
                self.num_heads,
                1,
                self.head_dim,
                dtype=v.dtype,
                device=v.device,
            )
            v = torch.cat([v, sink_values], dim=-2)

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

        if self.rope is not None:
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
