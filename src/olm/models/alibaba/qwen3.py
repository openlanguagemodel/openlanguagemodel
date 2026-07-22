from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUMoEFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class Qwen3Block(Block):
    """
    A single decoder block for Qwen3 (the flagship all-attention MoE, not
    the Qwen3-Next / Qwen3.5 hybrid-linear-attention variants).

    Every layer is identical: grouped-query attention with QK-norm, paired
    with a fine-grained MoE feed-forward (no shared experts, standard
    softmax top-k routing):

        x = x + GQA(RMSNorm(x))
        x = x + MoE(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        moe_intermediate_size (int): Hidden dimension of each MoE expert.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key/value heads (GQA).
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts routed to per token.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
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
        rms_norm_eps: float,
    ):
        super().__init__(
            [
                Residual(
                    Block(
                        [
                            RMSNorm(embed_dim, eps=rms_norm_eps),
                            GroupedQueryAttention(
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
                            ),
                        ]
                    )
                ),
                Residual(
                    Block(
                        [
                            RMSNorm(embed_dim, eps=rms_norm_eps),
                            SwiGLUMoEFFN(
                                embed_dim,
                                num_experts=num_experts,
                                num_shared_experts=0,
                                top_k=top_k,
                                hidden_dim=moe_intermediate_size,
                                bias=False,
                            ),
                        ]
                    )
                ),
            ]
        )


class Qwen3Model(Block):
    """
    Base class for the Qwen3 (flagship MoE) language model.

    Structure:
        Embedding -> [Qwen3Block] x N -> RMSNorm -> OutputHead.

    Every layer uses the same grouped-query attention (with QK-norm) and
    fine-grained MoE feed-forward (no shared experts, softmax top-k
    routing) -- unlike Qwen3-Next / Qwen3.5, there is no hybrid linear
    attention and no dense-first layers. Qwen3-235B-A22B does not tie
    input/output embeddings.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits
        shaped ``[batch, seq_len, vocab_size]``.

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
        rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: Qwen3Block(
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


class Qwen3_235B_A22B(Qwen3Model):
    """Qwen3-235B-A22B (235B total, ~22B active)."""

    def __init__(self):
        super().__init__(
            vocab_size=151936,
            embed_dim=4096,
            moe_intermediate_size=1536,
            num_layers=94,
            num_heads=64,
            num_kv_heads=4,
            max_seq_len=32768,
            num_experts=128,
            top_k=8,
            head_dim=128,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_weights=False,
        )
