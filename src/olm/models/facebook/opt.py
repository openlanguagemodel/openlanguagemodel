
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import MultiHeadAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.activations import ReLU
from olm.nn.norms import RMSNorm, LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.absolute import AbsolutePositionalEmbedding

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
        super().__init__([
            Residual(
                Block(
                    [
                        LayerNorm(embed_dim, eps=1e-6),
                        MultiHeadAttention(
                            embed_dim,
                            num_heads,
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
        ])
        
class OPTModel(Block):
    """
    OPT Model Definition.

    Implements a decoder-only Transformer with specific OPT optimizations:
    - Pre-normalization with LayerNorm
    - Multi-Head Attention with Causal Masking
    - ReLU activation in Feed-Forward Networks

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of heads.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, dropout=0.1):
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        super().__init__([
            self.token_embedding,
            AbsolutePositionalEmbedding(max_seq_len=2048, embed_dim=embed_dim, dropout=0.0),
            nn.Dropout(dropout),
            Repeat(lambda: OPTBlock(embed_dim, intermediate_size, num_heads, dropout), num_layers),
            LayerNorm(embed_dim, eps=1e-5),
            self.lm_head,
        ])

        # tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight

class OPT125M(OPTModel):
    """
    OPT 125M Model Definition.
    """

    def __init__(self):

        super().__init__(
            vocab_size = 50272,
            embed_dim = 768,
            intermediate_size = 3072,
            num_layers = 12,
            num_heads = 12,
            dropout = 0.1,
        )

        token_emb = self.blocks[0]     # Embedding wrapper
        lm_head = self.blocks[-1]      # nn.Linear

        lm_head.weight = token_emb.embedding.weight