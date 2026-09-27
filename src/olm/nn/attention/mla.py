import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.attention.base import AttentionwithRoPEBase
from olm.nn.torch_nn_wrappers import Linear
from olm.nn.embeddings.positional.rope import RotaryPositionalEmbedding
from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.norms import RMSNorm


def _apply_rope_layout(
    rope: nn.Module, x: torch.Tensor, interleaved: bool
) -> torch.Tensor:
    """
    Apply an interleaved-pair RoPE module in either rotary dimension layout.

    OLM's RoPE modules rotate adjacent pairs ``(x[2i], x[2i + 1])``. Checkpoints
    in the split-half ("rotate_half") layout pair ``x[i]`` with
    ``x[i + d/2]`` instead. For that layout the halves are interleaved, rotated,
    and split back, which is exactly the rotate-half rotation.

    Args:
        rope (nn.Module): RoPE module taking ``[batch, seq, heads, dim]``.
        x (torch.Tensor): Tensor shaped ``[batch, seq, heads, dim]``.
        interleaved (bool): True for adjacent pairs, False for split halves.

    Returns:
        torch.Tensor: The rotated tensor, same shape as ``x``.
    """
    if interleaved:
        return rope(x)

    half = x.shape[-1] // 2
    paired = torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)
    paired = rope(paired).view(*x.shape[:-1], half, 2)
    return torch.cat((paired[..., 0], paired[..., 1]), dim=-1)


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
        rope_interleaved (bool, optional): Rotary dimension layout of the
            ``qk_rope_head_dim`` slice. True pairs adjacent dimensions
            ``(2i, 2i + 1)`` (OLM's native layout); False pairs ``i`` with
            ``i + qk_rope_head_dim / 2`` (the "rotate_half" layout). Defaults
            to True.

    Forward:
        Takes ``x`` shaped ``[batch, seq_len, embed_dim]`` and an optional
        ``mask`` broadcastable to ``[batch, heads, seq_len, seq_len]``.
        Returns ``[batch, seq_len, embed_dim]``.
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
        rope_interleaved: bool = True,
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
        self.rope_interleaved = rope_interleaved

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
            self.q_b_proj = Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=bias)

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
        q, k, value, _ = self._project_qkv(x)
        attention_out = self.compute_attention(q, k, value, mask)

        # [B, H, N, v_head_dim] -> [B, N, H * v_head_dim]
        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(B, N, self.num_heads * self.v_head_dim)
        )

        return self.out_proj(attention_out)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project ``x`` to per-head queries, keys and values.

        Returns ``(q, k, value, q_latent)`` where ``q`` and ``k`` are
        ``[batch, heads, seq, qk_head_dim]``, ``value`` is
        ``[batch, heads, seq, v_head_dim]``, and ``q_latent`` is the input to
        ``q_b_proj`` (the normalized query latent) when ``q_lora_rank`` is set,
        or ``x`` itself otherwise. ``DeepSeekSparseAttention`` feeds
        ``q_latent`` to its lightning indexer.
        """
        B, N, _ = x.shape

        # --- Queries ---
        if self.q_lora_rank is None:
            q_latent = x
            q = self.q_proj(x)
        else:
            q_latent = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(q_latent)
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
        q_rope = self._apply_rope(q_rope)
        k_rope = self._apply_rope(k_rope.view(B, N, 1, self.qk_rope_head_dim))
        k_rope = k_rope.expand(B, N, self.num_heads, self.qk_rope_head_dim)

        # --- Assemble full query/key per head, then [B, N, H, D] -> [B, H, N, D] ---
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)
        return q, k, value, q_latent

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE using interleaved or split-half dimension pairs."""
        return _apply_rope_layout(self.rope, x, self.rope_interleaved)
