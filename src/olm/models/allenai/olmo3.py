import torch

from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Olmo3Block(Block):
    """
    A single Transformer block for OLMo 3 (OLMo 2 lineage).

    Uses the "reordered" post-norm structure: there is no input
    normalization; RMSNorm is applied to each sublayer's output before the
    residual add. Queries and keys are normalized (QK-norm). Attention is
    grouped-query (GQA reduces to MHA when ``num_kv_heads == num_heads``).

    Structure:
        x = x + post_attn_norm(GQA(x))
        x = x + post_ff_norm(SwiGLU(x))

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads.
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for RoPE cache).
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        sliding_window (int | None): Local attention window; ``None`` for a
            full-attention (global) layer.
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
        rope_theta: float,
        rms_norm_eps: float,
        sliding_window: int | None = 4096,
    ):
        super().__init__([])
        self.sliding_window = sliding_window
        # OLMo normalizes the full q/k projection; GQA applies per-head QK-norm,
        # which is used here as a close approximation.
        self.self_attn = GroupedQueryAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            head_dim=head_dim,
            dropout=dropout,
            rope_theta=rope_theta,
            use_bias=False,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.mlp = SwiGLUFFN(
            embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False
        )
        self.post_feedforward_layernorm = RMSNorm(embed_dim, eps=rms_norm_eps)

    def forward(self, x):
        residual = x
        x = self.self_attn(x, self._sliding_window_mask(x))
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x

    def _sliding_window_mask(self, x):
        if self.sliding_window is None:
            return None

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return positions[None, :] > positions[:, None] - self.sliding_window


class Olmo3Model(Block):
    """
    Base class for OLMo 3 models.

    Structure:
        Embedding -> [Olmo3Block] x N -> RMSNorm -> OutputHead.

    Attention alternates a 3:1 pattern of sliding-window ("local") layers
    followed by one full-attention ("global") layer. OLMo 3 does not tie
    input/output embeddings; the named presets pass ``tie_weights=False``.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

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
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        sliding_window (int): Local attention window size.
        sliding_window_pattern (int): Period of the sliding:full pattern; every
            ``sliding_window_pattern``-th layer is a full-attention layer.
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
        max_seq_len: int = 65536,
        rope_theta: float = 500000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 4096,
        sliding_window_pattern: int = 4,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: Olmo3Block(
                        embed_dim,
                        intermediate_size,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        dropout,
                        rope_theta,
                        rms_norm_eps,
                        sliding_window,
                    ),
                    num_layers,
                ),
                RMSNorm(embed_dim, eps=rms_norm_eps),
                OutputHead(
                    embed_dim,
                    vocab_size,
                    tied_embedding=embedding,
                    tie_weights=tie_weights,
                    use_norm=False,
                ),
            ]
        )

        # 3 sliding-window layers followed by 1 full-attention (global) layer.
        for layer_idx, block in enumerate(self.blocks[1].stack):
            is_global = layer_idx % sliding_window_pattern == sliding_window_pattern - 1
            block.sliding_window = None if is_global else sliding_window


class Olmo3_7B(Olmo3Model):
    """OLMo 3 7B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=100278,
            embed_dim=4096,
            intermediate_size=11008,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            max_seq_len=65536,
            rope_theta=500000.0,
            rms_norm_eps=1e-6,
            sliding_window=4096,
            tie_weights=False,
        )


class Olmo3_32B(Olmo3Model):
    """OLMo 3 32B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=100278,
            embed_dim=5120,
            intermediate_size=27648,
            num_layers=64,
            num_heads=40,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=65536,
            rope_theta=500000.0,
            rms_norm_eps=1e-6,
            sliding_window=4096,
            tie_weights=False,
        )
