from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import FlashAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms.layer_norm import LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead


class OLMoBlock(Block):
    """
    A single Transformer block for the OLMo architecture.

    Structure:
        x = x + Attn(LN(x))
        x = x + SwiGLU(LN(x))

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Max context.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__(
            [
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim, elementwise_affine=False),
                            FlashAttentionwithRoPE(
                                embed_dim,
                                num_heads,
                                max_seq_len,
                                dropout=dropout,
                                causal=True,
                                bias=False,  # No bias for OLMo
                            ),
                        ]
                    )
                ),
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim, elementwise_affine=False),
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


class OLMoModel(Block):
    """
    Base class for the OLMo (Open Language Model) architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                embedding,
                Repeat(
                    lambda: OLMoBlock(
                        embed_dim, intermediate_size, num_heads, max_seq_len, dropout
                    ),
                    num_layers,
                ),
                LayerNorm(embed_dim, elementwise_affine=False),
                OutputHead(
                    embed_dim,
                    vocab_size,
                    tied_embedding=embedding,
                    tie_weights=tie_weights,
                    use_norm=False,
                ),
            ]
        )


class OLMo_7B(OLMoModel):
    """OLMo 7B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=50280,
            embed_dim=4096,
            intermediate_size=22016,
            num_layers=32,
            num_heads=32,
            max_seq_len=2048,
        )
