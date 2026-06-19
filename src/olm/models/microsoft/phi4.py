from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding


class Phi4Block(Block):
    """
    A single Transformer block for Phi 4.

    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Max sequence length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
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
    ):
        super().__init__(
            [
                Residual(
                    Block(
                        [
                            RMSNorm(embed_dim, eps=1e-5),
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
                            RMSNorm(embed_dim, eps=1e-5),
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


class Phi4Model(Block):
    """
    Base class for Phi 4 models.

    Implementation note:
        This implementation uses standard Rotary Positional Embeddings (RoPE)
        parameterized via `rope_theta`.
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
        rope_theta: float = 250000.0,
        dropout: float = 0.0,
        tie_weights: bool = False,
    ):
        super().__init__(
            [
                Embedding(vocab_size, embed_dim),
                Repeat(
                    lambda: Phi4Block(
                        embed_dim,
                        intermediate_size,
                        num_heads,
                        num_kv_heads,
                        max_seq_len,
                        dropout,
                        rope_theta,
                    ),
                    num_layers,
                ),
                RMSNorm(embed_dim, eps=1e-5),
                Linear(
                    embed_dim, vocab_size, bias=False
                ),  # torch.nn.Linear can also be used
            ]
        )

        if tie_weights:
            self.blocks[3].weight = self.blocks[0].embedding.weight


class Phi4_14B(Phi4Model):
    """Phi-4 14B Model."""

    def __init__(self):
        super().__init__(
            vocab_size=100352,
            embed_dim=5120,
            intermediate_size=17920,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            max_seq_len=16384,
            rope_theta=250000.0,
        )
