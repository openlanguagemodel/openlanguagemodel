from __future__ import annotations

import math

import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat
from olm.nn.attention import GroupedQueryAttention
from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.feedforward import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Gemma2Embedding(Embedding):
    """Gemma 2 token embedding with hidden-size scaling."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)
        self.embed_scale = math.sqrt(embedding_dim)

    def forward(self, x):
        return super().forward(x) * self.embed_scale


class Gemma2FinalLogitSoftcap(nn.Module):
    """Gemma 2 final logit soft-capping."""

    def __init__(self, softcap: float | None = 30.0):
        super().__init__()
        self.softcap = softcap

    def forward(self, logits):
        if self.softcap is None:
            return logits
        return torch.tanh(logits / self.softcap) * self.softcap


class Gemma2Block(Block):
    """
    A single Transformer block for Gemma 2.

    Implements the "Sandwich" Normalization pattern:
    Norm -> Attn -> Norm -> Residual
    Norm -> MLP  -> Norm -> Residual

    Structure:
        x = x + post_attn_norm(Attn(input_norm(x)))
        x = x + post_ffn_norm(GeGLU(pre_ffn_norm(x)))

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
        head_dim (int): Dimension per attention head.
        sliding_window (int | None): Local attention window; ``None`` for a
            global (full-attention) layer.
        attn_logit_softcap (float | None): Tanh soft-cap on attention logits.
        query_pre_attn_scalar (float | None): Attention logits are scaled by
            ``query_pre_attn_scalar ** -0.5``; ``None`` uses ``head_dim``.

    Forward:
        Accepts hidden states shaped ``[batch, seq_len, embed_dim]`` and an
        optional boolean ``mask`` (True means attend). The mask is combined
        with the causal mask and, on local layers, the sliding window.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float,
        rope_theta: float,
        head_dim: int,
        sliding_window: int | None = 4096,
        attn_logit_softcap: float | None = 50.0,
        query_pre_attn_scalar: float | None = 256.0,
    ):
        super().__init__([])
        self.sliding_window = sliding_window
        self.input_layernorm = RMSNorm(embed_dim, eps=1e-6)
        self.self_attn = GroupedQueryAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            head_dim=head_dim,
            dropout=dropout,
            rope_theta=rope_theta,
            use_bias=False,
            attention_scale=(
                query_pre_attn_scalar**-0.5
                if query_pre_attn_scalar is not None
                else None
            ),
            attn_logit_softcap=attn_logit_softcap,
        )
        self.post_attention_layernorm = RMSNorm(embed_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(embed_dim, eps=1e-6)
        self.mlp = GeGLUFFN(
            embed_dim,
            hidden_dim=intermediate_size,
            dropout=dropout,
            bias=False,
            approximate="tanh",
        )
        self.post_feedforward_layernorm = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, self._attention_mask(x, mask))
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x

    def _attention_mask(self, x, mask):
        """Combine the caller's ``mask`` with this layer's sliding window."""
        if self.sliding_window is None:
            return mask

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        window = positions[None, :] > positions[:, None] - self.sliding_window
        if mask is None:
            return window
        return attention_mask_to_bool(mask, device=x.device) & window


class Gemma2Model(Block):
    """
    Base class for Gemma 2 models.

    Structure:
        Scaled token embedding -> [Gemma2Block] x N -> RMSNorm ->
        tied OutputHead -> optional final logit softcap.

    Layers alternate local (sliding-window) and global attention, starting
    with a local layer. The feed-forward uses the tanh approximation of GELU
    (``gelu_pytorch_tanh``), as in the released checkpoints.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``. An optional boolean ``mask`` (True
        means attend) reaches every attention layer.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads.
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length.
        rope_theta (float): RoPE base frequency.
        dropout (float): Dropout probability.
        sliding_window (int | None): Window of the local layers.
        attn_logit_softcap (float | None): Tanh soft-cap on attention logits.
        final_logit_softcap (float | None): Tanh soft-cap on output logits.
        query_pre_attn_scalar (float | None): Attention logits are scaled by
            ``query_pre_attn_scalar ** -0.5``; ``None`` uses ``head_dim``.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        sliding_window: int | None = 4096,
        attn_logit_softcap: float | None = 50.0,
        final_logit_softcap: float | None = 30.0,
        query_pre_attn_scalar: float | None = 256.0,
        tie_weights: bool = True,
    ):
        embedding = Gemma2Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: Gemma2Block(
                        embed_dim,
                        intermediate_size,
                        num_heads,
                        num_kv_heads,
                        max_seq_len,
                        dropout,
                        rope_theta,
                        head_dim,
                        sliding_window=None,
                        attn_logit_softcap=attn_logit_softcap,
                        query_pre_attn_scalar=query_pre_attn_scalar,
                    ),
                    num_layers,
                ),
                RMSNorm(embed_dim, eps=1e-6),
                OutputHead(
                    embed_dim,
                    vocab_size,
                    tied_embedding=embedding,
                    tie_weights=tie_weights,
                    use_norm=False,
                ),
                Gemma2FinalLogitSoftcap(final_logit_softcap),
            ]
        )

        for layer_idx, block in enumerate(self.blocks[1].stack):
            block.sliding_window = sliding_window if layer_idx % 2 == 0 else None


class Gemma2_27B(Gemma2Model):
    """Gemma 2 27B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=4608,
            intermediate_size=36864,
            num_layers=46,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128,
            max_seq_len=8192,
            rope_theta=10000.0,
            sliding_window=4096,
            attn_logit_softcap=50.0,
            final_logit_softcap=30.0,
            query_pre_attn_scalar=144.0,
        )


class Gemma2_9B(Gemma2Model):
    """Gemma 2 9B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=3584,
            intermediate_size=14336,
            num_layers=42,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0,
            sliding_window=4096,
            attn_logit_softcap=50.0,
            final_logit_softcap=30.0,
            query_pre_attn_scalar=256.0,
        )


class Gemma2_2B(Gemma2Model):
    """Gemma 2 2B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=2304,
            intermediate_size=9216,
            num_layers=26,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0,
            sliding_window=4096,
            attn_logit_softcap=50.0,
            final_logit_softcap=30.0,
            query_pre_attn_scalar=256.0,
        )
