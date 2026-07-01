import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.norms import RMSNorm
from olm.nn.embeddings.positional.rope import RotaryPositionalEmbedding


class MultiHeadLatentAttention(AttentionwithRoPEBase):
    """
    Multi-head Latent Attention (MLA) as used by DeepSeek V3, Sarvam 105B,
    Ling 2.5, Kimi K2/K2.5, Mistral Large 3.

    MLA compresses key-value representations through a low-rank bottleneck
    (``kv_lora_rank``), dramatically reducing the KV cache.  Query and key
    heads are split into position-dependent (RoPE) and position-free (NoPE)
    components.

    Projection flow:
        - Q: ``x -> q_proj -> [q_nope, q_rope]`` (optionally via q_lora)
        - KV: ``x -> kv_proj (low rank) -> kv_norm -> kv_up_proj -> [k_nope, k_rope, v]``

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of query heads.
        qk_nope_head_dim: Non-positional dimensions per Q/K head.
        qk_rope_head_dim: Positional (RoPE) dimensions per Q/K head.
        v_head_dim: Value head dimension.
        kv_lora_rank: Low-rank dimension for KV compression.
        q_lora_rank: Optional low-rank dimension for Q compression.
            If 0 or None, Q is projected directly.
        max_seq_len: Maximum sequence length (for RoPE).
        rope_theta: Base frequency for RoPE.
        use_qk_norm: Apply RMSNorm to compressed KV.
        rms_norm_eps: Epsilon for normalization.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        kv_lora_rank: int = 512,
        q_lora_rank: Optional[int] = None,
        max_seq_len: int = 131072,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank or 0
        self.dropout_p = dropout
        self.scale = self.qk_head_dim ** -0.5
        self.max_seq_len = max_seq_len

        # Q path
        if self.q_lora_rank > 0:
            self.q_down_proj = Linear(embed_dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=rms_norm_eps)
            self.q_up_proj = Linear(
                self.q_lora_rank, num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = Linear(embed_dim, num_heads * self.qk_head_dim, bias=False)

        # KV path (low-rank compression)
        self.kv_down_proj = Linear(embed_dim, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps) if use_qk_norm else nn.Identity()
        self.kv_up_proj = Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.k_rope_proj = Linear(embed_dim, qk_rope_head_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(
            qk_rope_head_dim, max_seq_len, base=rope_theta
        )

        self.out_proj = Linear(num_heads * v_head_dim, embed_dim, bias=False)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        # --- Q path ---
        if self.q_lora_rank > 0:
            q = self.q_up_proj(self.q_norm(self.q_down_proj(x)))
        else:
            q = self.q_proj(x)

        q = q.view(B, N, self.num_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]

        # --- KV path ---
        kv_compressed = self.kv_norm(self.kv_down_proj(x))
        kv = self.kv_up_proj(kv_compressed)
        kv = kv.view(B, N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k_rope = self.k_rope_proj(x).unsqueeze(2)
        k_rope = k_rope.expand(B, N, self.num_heads, self.qk_rope_head_dim)

        # --- Apply RoPE to positional components ---
        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope)

        # --- Concat nope + rope for full Q and K ---
        q_full = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k_full = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = self.compute_attention(q_full, k_full, v, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.v_head_dim)
        return self.out_proj(attn_out)
