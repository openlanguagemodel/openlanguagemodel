import torch
import torch.nn.functional as F
from typing import Optional

from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.torch_nn_wrappers import Linear
from olm.nn.embeddings.positional.rope import RotaryPositionalEmbedding
from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.norms import RMSNorm


class MultiHeadLatentAttention(AttentionwithRoPEBase):
    """
    Multi-head Latent Attention (MLA), introduced in DeepSeek-V2 and used by
    DeepSeek-V3/R1, Kimi K2, Mistral Large 3, and others.

    MLA compresses keys and values into a low-rank latent vector that is the only
    thing that needs to be cached at inference time, drastically shrinking the KV
    cache. Positional information is handled by a *decoupled* RoPE: each head's
    query/key is split into a "nope" part (no positional encoding, carried through
    the compressed latent) and a small "rope" part that receives rotary embeddings.
    The rope part of the key is shared across all heads (MQA-style).

    Reference:
        "DeepSeek-V2" (DeepSeek-AI, 2024), arXiv:2405.04434

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length (for the RoPE cache).
        kv_lora_rank (int): Rank of the compressed key/value latent.
        qk_nope_head_dim (int): Per-head dimension of the non-positional query/key part.
        qk_rope_head_dim (int): Per-head dimension of the rotary query/key part.
        v_head_dim (int): Per-head dimension of the value.
        q_lora_rank (int, optional): Rank of the compressed query latent. If
            ``None``, queries are projected directly without low-rank compression
            (as in DeepSeek-V2-Lite). Defaults to None.
        dropout (float, optional): Attention dropout probability. Defaults to 0.0.
        causal (bool, optional): If True, applies causal masking for autoregressive
            models. Defaults to True.
        rope_theta (float, optional): RoPE base frequency. Defaults to 10000.0.
        bias (bool, optional): Whether to use bias in the linear projections.
            Defaults to False.
        rms_norm_eps (float, optional): Epsilon for the latent RMSNorm layers.
            Defaults to 1e-6.
        attention_scale (float, optional): Override for the softmax scale. Defaults
            to ``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = True,
        rope_theta: float = 10000.0,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
        attention_scale: Optional[float] = None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            max_seq_len,
            dropout=dropout,
            bias=bias,
            rope_theta=rope_theta,
        )
        self.causal = causal
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Query and key share this per-head dimension when computing scores.
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scale = (
            attention_scale if attention_scale is not None else self.qk_head_dim**-0.5
        )
        self.dropout_p = dropout

        # AttentionwithRoPEBase sets up the shared scaffolding (dropout, scale, and
        # SDPA path via compute_attention), but its symmetric embed_dim->embed_dim
        # q/k/v projections and full-width RoPE do not fit MLA's low-rank latent
        # structure. Replace them with MLA's projections and a decoupled RoPE that
        # acts only on the rope sub-dimension of each head.
        del self.q_proj, self.k_proj, self.v_proj
        self.rope = RotaryPositionalEmbedding(
            qk_rope_head_dim, max_seq_len, base=rope_theta
        )

        # Query projection: optionally low-rank (down -> norm -> up).
        if q_lora_rank is None:
            self.q_proj = Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias)
        else:
            self.q_a_proj = Linear(embed_dim, q_lora_rank, bias=bias)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = Linear(
                q_lora_rank, num_heads * self.qk_head_dim, bias=bias
            )

        # Key/value projection: down to a shared latent plus the decoupled rope key.
        self.kv_a_proj_with_mqa = Linear(
            embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )

        self.out_proj = Linear(num_heads * v_head_dim, embed_dim, bias=bias)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention over the assembled MLA heads.

        Mirrors ``FlashAttentionwithRoPE`` by delegating to PyTorch's
        ``scaled_dot_product_attention`` (Flash Attention 2 when available) and
        folding causality into the supplied mask when both are present.

        Args:
            q (torch.Tensor): Query tensor ``[batch, heads, seq, qk_head_dim]``.
            k (torch.Tensor): Key tensor ``[batch, heads, seq, qk_head_dim]``.
            v (torch.Tensor): Value tensor ``[batch, heads, seq, v_head_dim]``.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Attention output ``[batch, heads, seq, v_head_dim]``.
        """
        attn_mask = None
        is_causal = self.causal
        if mask is not None:
            attn_mask = attention_mask_to_bool(mask, device=q.device)
            if self.causal:
                L_q, L_k = q.size(-2), k.size(-2)
                causal_bool = torch.ones(
                    L_q, L_k, device=q.device, dtype=torch.bool
                ).tril()
                attn_mask = attn_mask & causal_bool
                is_causal = False

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Multi-head Latent Attention.

        Args:
            x (torch.Tensor): Input tensor of shape ``[batch, seq_len, embed_dim]``.
            mask (torch.Tensor, optional): Attention mask broadcastable to
                ``[batch, heads, seq_len, seq_len]``. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch, seq_len, embed_dim]``.
        """
        B, N, _ = x.shape

        # --- Queries ---
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, N, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # --- Keys / Values ---
        kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_rope = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(B, N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # --- Decoupled RoPE ---
        # Query rope part is per-head; key rope part is shared across heads (MQA).
        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope.view(B, N, 1, self.qk_rope_head_dim))
        k_rope = k_rope.expand(B, N, self.num_heads, self.qk_rope_head_dim)

        # --- Assemble full query/key per head, then [B, N, H, D] -> [B, H, N, D] ---
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)

        attention_out = self.compute_attention(q, k, value, mask)

        # [B, H, N, v_head_dim] -> [B, N, H * v_head_dim]
        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(B, N, self.num_heads * self.v_head_dim)
        )

        return self.out_proj(attention_out)
