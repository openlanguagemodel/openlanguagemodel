import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import FlashAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.activations import ReLU
from olm.nn.norms import LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.absolute import AbsolutePositionalEmbedding
from olm.nn.blocks import OutputHead


class OPTBlock(Block):
    """
    A single Transformer block for the OPT architecture.

    Composes a Residual Multi-Head Attention block and a Residual ReLU
    Feed-Forward block, both utilizing Pre-LayerNorm.

    Structure:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + ReLU(LayerNorm(x))
    Args:
        embed_dim (int): The dimension of the model.
        intermediate_size (int): The hidden dimension of the feed-forward network.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__(
            [
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim, eps=1e-6),
                            FlashAttention(
                                embed_dim,
                                num_heads,
                                dropout=dropout,
                                causal=True,
                            ),
                        ]
                    )
                ),
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim, eps=1e-6),
                            ClassicFFN(
                                embed_dim,
                                hidden_dim=intermediate_size,
                                dropout=dropout,
                                activation_fn=ReLU(),
                            ),
                        ]
                    )
                ),
            ]
        )


class OPTModel(Block):
    """
    OPT Model Definition.

    Implements a decoder-only Transformer with specific OPT optimizations:
    - Pre-normalization with LayerNorm
    - Multi-Head Attention with Causal Masking
    - ReLU activation in Feed-Forward Networks
    - Tied output projection through ``OutputHead`` by default

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of heads.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        intermediate_size,
        num_layers,
        num_heads,
        dropout=0.1,
        tie_weights=True,
    ):
        token_embedding = Embedding(vocab_size, embed_dim)
        lm_head = OutputHead(
            embed_dim,
            vocab_size,
            tied_embedding=token_embedding,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__(
            [
                token_embedding,
                AbsolutePositionalEmbedding(
                    max_seq_len=2048, embed_dim=embed_dim, dropout=0.0
                ),
                nn.Dropout(dropout),
                Repeat(
                    lambda: OPTBlock(embed_dim, intermediate_size, num_heads, dropout),
                    num_layers,
                ),
                LayerNorm(embed_dim, eps=1e-5),
                lm_head,
            ]
        )

        self.token_embedding = token_embedding
        self.lm_head = lm_head


class OPT125M(OPTModel):
    """
    OPT 125M Model Definition.
    """

    def __init__(self):
        super().__init__(
            vocab_size=50272,
            embed_dim=768,
            intermediate_size=3072,
            num_layers=12,
            num_heads=12,
            dropout=0.1,
        )
