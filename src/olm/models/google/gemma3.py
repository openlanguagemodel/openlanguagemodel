from __future__ import annotations

import math

import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention, SlidingWindowAttention
from olm.nn.feedforward import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Gemma3Embedding(Embedding):
    """Gemma 3 token embedding with hidden-size scaling (same as Gemma 2)."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)
        self.embed_scale = math.sqrt(embedding_dim)

    def forward(self, x):
        return super().forward(x) * self.embed_scale


class Gemma3Block(Block):
    """
    A single Transformer block for Gemma 3.

    Implements the "sandwich" normalization pattern carried over from
    Gemma 2 (Norm -> sublayer -> Norm -> residual), but attention alternates
    between local (sliding-window) and global (full) layers with two
    independent RoPE base frequencies -- 10k for local, 1M for global --
    rather than a single shared attention module with a masked-out window.
    Unlike Gemma 2, there is no attention-logit softcapping; QK-norm is used
    instead to stabilize attention at scale.

    Structure:
        x = x + post_attn_norm(Attn(input_norm(x)))
        x = x + post_ffn_norm(GeGLU(pre_ffn_norm(x)))
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dropout: float,
        is_local: bool,
        sliding_window: int,
        local_rope_theta: float,
        global_rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.input_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)
        if is_local:
            self.self_attn = SlidingWindowAttention(
                embed_dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                window_size=sliding_window,
                head_dim=head_dim,
                dropout=dropout,
                rope_theta=local_rope_theta,
                use_qk_norm=True,
                rms_norm_eps=rms_norm_eps,
                use_bias=False,
            )
        else:
            self.self_attn = GroupedQueryAttention(
                embed_dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                head_dim=head_dim,
                dropout=dropout,
                rope_theta=global_rope_theta,
                use_qk_norm=True,
                rms_norm_eps=rms_norm_eps,
                use_bias=False,
            )
        self.post_attention_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.mlp = GeGLUFFN(
            embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False
        )
        self.post_feedforward_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x


class Gemma3Model(Block):
    """
    Base class for Gemma 3 models (text-only language-model component).

    Structure:
        Scaled token embedding -> [Gemma3Block] x N -> RMSNorm ->
        tied OutputHead.

    Attention alternates a 5:1 pattern of local sliding-window layers
    followed by one global full-attention layer (``sliding_window_pattern``
    controls the period). Local layers use ``local_rope_theta`` (10k);
    global layers use ``global_rope_theta`` (1M). Gemma 3 does not use
    final-logit softcapping (removed relative to Gemma 2, replaced by
    QK-norm). The vision tower present in the multimodal checkpoints is
    omitted; this models the text-only language-model component.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits
        shaped ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads.
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length.
        local_rope_theta (float): RoPE base frequency for local layers.
        global_rope_theta (float): RoPE base frequency for global layers.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        sliding_window (int): Local attention window size.
        sliding_window_pattern (int): Period of the local:global pattern;
            every ``sliding_window_pattern``-th layer is global.
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
        local_rope_theta: float = 10000.0,
        global_rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 1024,
        sliding_window_pattern: int = 6,
        tie_weights: bool = True,
    ):
        embedding = Gemma3Embedding(vocab_size, embed_dim)

        # 5 local (sliding-window) layers followed by 1 global (full) layer.
        layers = nn.ModuleList(
            [
                Gemma3Block(
                    embed_dim,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    max_seq_len,
                    dropout,
                    is_local=(
                        layer_idx % sliding_window_pattern != sliding_window_pattern - 1
                    ),
                    sliding_window=sliding_window,
                    local_rope_theta=local_rope_theta,
                    global_rope_theta=global_rope_theta,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim,
            vocab_size,
            tied_embedding=embedding,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x):
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class Gemma3_4B(Gemma3Model):
    """Gemma 3 4B Model (text-only language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=262144,
            embed_dim=2560,
            intermediate_size=10240,
            num_layers=34,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            max_seq_len=131072,
            sliding_window=1024,
            sliding_window_pattern=6,
        )


class Gemma3_27B(Gemma3Model):
    """Gemma 3 27B Model (text-only language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=262144,
            embed_dim=5376,
            intermediate_size=21504,
            num_layers=62,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128,
            max_seq_len=131072,
            sliding_window=1024,
            sliding_window_pattern=6,
        )
