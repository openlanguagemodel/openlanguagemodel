from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class MistralSmall3_1_Block(Block):
    """
    A single decoder block for Mistral Small 3.1.

    Plain dense grouped-query attention (no sliding window -- dropped after
    the original Mistral 7B) and a SwiGLU feed-forward, pre-norm:

        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Max sequence length (for the RoPE cache).
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
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
                                dropout=dropout,
                                rope_theta=rope_theta,
                                use_bias=False,
                            ),
                        ]
                    )
                ),
                Residual(
                    Block(
                        [
                            RMSNorm(embed_dim, eps=rms_norm_eps),
                            SwiGLUFFN(
                                embed_dim,
                                hidden_dim=intermediate_size,
                                dropout=dropout,
                                bias=False,
                            ),
                        ]
                    )
                ),
            ]
        )


class MistralSmall3_1_Model(Block):
    """
    Base class for the Mistral Small 3.1 language model.

    Structure:
        Embedding -> [MistralSmall3_1_Block] x N -> RMSNorm -> OutputHead.

    A plain dense (non-MoE) grouped-query-attention transformer -- unlike
    Mistral Large 3, Small 3.1 uses neither Multi-head Latent Attention nor
    a mixture of experts. Mistral Small 3.1 does not tie input/output
    embeddings.

    Notes:
        - This models the language-model component only; the reference
          checkpoint also includes a vision encoder, which is omitted here.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits
        shaped ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Maximum context length.
        rope_theta (float): RoPE base frequency.
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
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
        max_seq_len: int,
        rope_theta: float = 1000000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: MistralSmall3_1_Block(
                        embed_dim,
                        intermediate_size,
                        num_heads,
                        num_kv_heads,
                        max_seq_len,
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


class MistralSmall3_1_24B(MistralSmall3_1_Model):
    """Mistral Small 3.1 (24B; language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=131072,
            embed_dim=5120,
            intermediate_size=32768,
            num_layers=40,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=1000000000.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )
