import torch
import torch.nn as nn
from olm.nn.torch_nn_wrappers import Linear
import torch.nn.functional as F
from typing import Optional
from olm.nn.embeddings.positional.rope import (
    PartialRotaryPositionalEmbedding,
    RotaryPositionalEmbedding,
)
from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.norms import RMSNorm


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Positional Embeddings.

    GQA is a distinct attention mechanism where the number of Key/Value heads is smaller
    than the number of Query heads. This reduces memory bandwidth usage during inference
    (smaller KV cache) while maintaining performance close to Multi-Head Attention (MHA).

    If num_kv_heads == num_heads, this is equivalent to MHA.
    If num_kv_heads == 1, this is equivalent to Multi-Query Attention (MQA).

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of Query heads.
        num_kv_heads (int): Number of Key/Value heads. Must divide num_heads.
        max_seq_len (int): Maximum sequence length for RoPE.
        head_dim (int, optional): Per-head query/key dimension. Defaults to
            ``embed_dim // num_heads``.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        rope_theta (float, optional): Base frequency for RoPE. Defaults to 10000.0.
        use_bias (bool, optional): Whether the output projection has a bias.
            Defaults to False.
        qkv_bias (bool, optional): Whether the query/key/value projections have
            a bias (Qwen 2 style). Defaults to False.
        use_qk_norm (bool, optional): Apply per-head RMSNorm to queries and keys
            before RoPE. Defaults to False.
        rms_norm_eps (float, optional): Epsilon for the QK-norm layers.
            Defaults to 1e-6.
        attention_scale (float, optional): Softmax scale. Defaults to
            ``head_dim ** -0.5``.
        attn_logit_softcap (float, optional): If set, attention logits are
            soft-capped with ``cap * tanh(logits / cap)`` (Gemma 2). Defaults
            to None.
        use_rope (bool, optional): Whether to apply rotary embeddings. Set to
            False for NoPE layers. Defaults to True.
        partial_rotary_factor (float, optional): Fraction of ``head_dim`` that
            receives RoPE. Defaults to 1.0.
        use_attention_sink (bool, optional): Add one learnable sink logit per
            head that absorbs attention mass. Defaults to False.
        v_head_dim (int, optional): Per-head value dimension, for architectures
            whose values are narrower or wider than their queries and keys.
            Defaults to ``head_dim``.
        value_scale (float, optional): Constant multiplier applied to the
            projected values before attention. Defaults to None (no scaling).

    Forward:
        Takes ``x`` shaped ``[batch, seq_len, embed_dim]`` and an optional
        boolean/binary/additive ``mask`` (True or 1 means attend), which is
        combined with the causal mask. Returns ``[batch, seq_len, embed_dim]``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        attention_scale: Optional[float] = None,
        attn_logit_softcap: Optional[float] = None,
        use_rope: bool = True,
        partial_rotary_factor: float = 1.0,
        use_attention_sink: bool = False,
        v_head_dim: Optional[int] = None,
        value_scale: Optional[float] = None,
    ):
        super().__init__()

        # If head_dim is not provided, assume standard division
        if head_dim is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = head_dim

        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        if self.v_head_dim < 1:
            raise ValueError(f"v_head_dim must be positive, got {self.v_head_dim}")
        self.value_scale = value_scale

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads

        self.q_dim = num_heads * self.head_dim
        self.kv_dim = num_kv_heads * self.head_dim
        self.v_dim = num_kv_heads * self.v_head_dim

        self.dropout_p = dropout
        self.scale = (
            attention_scale if attention_scale is not None else self.head_dim**-0.5
        )
        self.attn_logit_softcap = attn_logit_softcap
        self.use_rope = use_rope
        self.use_attention_sink = use_attention_sink

        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                "partial_rotary_factor must be in (0.0, 1.0], "
                f"got {partial_rotary_factor}"
            )

        # QK Norm (Qwen 2/2.5 feature)
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

        # Projections
        self.q_proj = Linear(
            embed_dim, self.q_dim, bias=qkv_bias
        )  # torch.nn.Linear can also be used
        self.k_proj = Linear(
            embed_dim, self.kv_dim, bias=qkv_bias
        )  # torch.nn.Linear can also be used
        self.v_proj = Linear(
            embed_dim, self.v_dim, bias=qkv_bias
        )  # torch.nn.Linear can also be used
        self.out_proj = Linear(
            num_heads * self.v_head_dim, embed_dim, bias=use_bias
        )  # torch.nn.Linear can also be used

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Grouped Query Attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, embed_dim].
            mask (torch.Tensor, optional): Attention mask of shape [batch, 1, seq_len, seq_len]
                or [batch, seq_len, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, embed_dim].
        """
        B, N, C = x.shape

        # Project projections
        # q: [B, N, num_heads * head_dim]
        # k: [B, N, num_kv_heads * head_dim]
        # v: [B, N, num_kv_heads * v_head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, N, Heads, D]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_kv_heads, self.head_dim)
        v = v.view(B, N, self.num_kv_heads, self.v_head_dim)
        if self.value_scale is not None:
            v = v * self.value_scale

        # Apply QK Norm if enabled (per head, before RoPE)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Transpose for attention: [B, Heads, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand K and V to match num_heads if needed (GQA replication)
        if self.num_groups > 1:
            k = k[:, :, None, :, :].expand(
                B, self.num_kv_heads, self.num_groups, N, self.head_dim
            )
            k = k.reshape(B, self.num_heads, N, self.head_dim)

            v = v[:, :, None, :, :].expand(
                B, self.num_kv_heads, self.num_groups, N, self.v_head_dim
            )
            v = v.reshape(B, self.num_heads, N, self.v_head_dim)

        attn_mask = None
        is_causal = True
        if mask is not None:
            attn_mask = attention_mask_to_bool(mask, device=x.device)
            causal_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
            attn_mask = attn_mask & causal_mask
            is_causal = False

        if self.attn_logit_softcap is None and not self.use_attention_sink:
            # Scaled Dot Product Attention
            # Uses Flash Attention optimization if available via F.scaled_dot_product_attention
            attention_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.attn_logit_softcap is not None:
                softcap = self.attn_logit_softcap
                attn_scores = torch.tanh(attn_scores / softcap) * softcap

            if is_causal:
                causal_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
                attn_mask = (
                    causal_mask if attn_mask is None else attn_mask & causal_mask
                )

            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

            if self.use_attention_sink:
                sink_scores = self.attention_sink.view(1, self.num_heads, 1, 1)
                sink_scores = sink_scores.expand(B, self.num_heads, N, 1)
                attn_scores = torch.cat([attn_scores, sink_scores], dim=-1)
                sink_values = torch.zeros(
                    B,
                    self.num_heads,
                    1,
                    self.v_head_dim,
                    dtype=v.dtype,
                    device=v.device,
                )
                values = torch.cat([v, sink_values], dim=-2)
            else:
                values = v

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attention_out = torch.matmul(attn_probs, values)

        # [B, Heads, N, D] -> [B, N, Heads, D] -> [B, N, num_heads * v_head_dim]
        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(B, N, self.num_heads * self.v_head_dim)
        )

        return self.out_proj(attention_out)
