from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUMoEFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.rope import PartialRotaryPositionalEmbedding
from olm.nn.blocks import OutputHead


class MiniMaxM2Block(Block):
    """A single Transformer block for MiniMax-M2.

    Structure:
        x = x + GQA(RMSNorm(x))      # QK-norm + partial RoPE, full causal attention
        x = x + MoE(RMSNorm(x))      # sigmoid-routed SwiGLU experts

    Args:
        embed_dim (int): Model dimension.
        moe_intermediate_size (int): Hidden dimension of each MoE expert.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (GQA).
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for RoPE cache).
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts routed to per token.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
        rotary_percentage (float): Fraction of ``head_dim`` rotated by RoPE.
        use_qk_norm (bool): Apply per-head RMSNorm to queries and keys.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        embed_dim: int,
        moe_intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        rope_theta: float,
        rotary_percentage: float,
        use_qk_norm: bool,
        rms_norm_eps: float,
    ):
        attn = GroupedQueryAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            head_dim=head_dim,
            dropout=dropout,
            rope_theta=rope_theta,
            use_bias=False,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=rms_norm_eps,
        )
        # MiniMax-M2 rotates only a fraction of each head (config rotary_dim=64,
        # head_dim=128 -> 0.5). Swap the full-RoPE module GQA builds by default
        # for a partial one with the same forward signature.
        attn.rope = PartialRotaryPositionalEmbedding(
            head_dim,
            rotary_percentage=rotary_percentage,
            base=rope_theta,
            max_seq_len=max_seq_len,
        )

        # MiniMax-M2 scores experts with sigmoid rather than softmax
        # (``scoring_func='sigmoid'``).
        moe = SwiGLUMoEFFN(
            embed_dim,
            num_experts=num_experts,
            num_shared_experts=0,
            top_k=top_k,
            hidden_dim=moe_intermediate_size,
            bias=False,
            router_kwargs={"scoring_func": "sigmoid"},
        )

        super().__init__(
            [
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), attn])),
                Residual(Block([RMSNorm(embed_dim, eps=rms_norm_eps), moe])),
            ]
        )


class MiniMaxM2Model(Block):
    """Base class for MiniMax-M2 models.

    Structure:
        Embedding -> [MiniMaxM2Block] x N -> RMSNorm -> OutputHead.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

    Notes:
        - MiniMax-M2 does **not** tie input/output embeddings; the named
          ``MiniMaxM2`` preset passes ``tie_weights=False``.
        - The Multi-Token-Prediction head present in the reference checkpoint is
          a training/inference feature and is intentionally omitted here.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        moe_intermediate_size (int): Hidden dimension of each MoE expert.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (GQA).
        max_seq_len (int): Maximum context length.
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts routed to per token.
        head_dim (int): Dimension per attention head.
        rope_theta (float): RoPE base frequency.
        rotary_percentage (float): Fraction of ``head_dim`` rotated by RoPE.
        use_qk_norm (bool): Apply per-head RMSNorm to queries and keys.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        moe_intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        num_experts: int,
        top_k: int,
        head_dim: int = 128,
        rope_theta: float = 5000000.0,
        rotary_percentage: float = 0.5,
        use_qk_norm: bool = True,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: MiniMaxM2Block(
                        embed_dim,
                        moe_intermediate_size,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        max_seq_len,
                        num_experts,
                        top_k,
                        dropout,
                        rope_theta,
                        rotary_percentage,
                        use_qk_norm,
                        rms_norm_eps,
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


class MiniMaxM2(MiniMaxM2Model):
    """MiniMax-M2 (230B total, 256-expert MoE, ~10B active per token)."""

    def __init__(self):
        super().__init__(
            vocab_size=200064,
            embed_dim=3072,
            moe_intermediate_size=1536,
            num_layers=62,
            num_heads=48,
            num_kv_heads=8,
            max_seq_len=196608,
            num_experts=256,
            top_k=8,
            head_dim=128,
            rope_theta=5000000.0,
            rotary_percentage=0.5,
            use_qk_norm=True,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
