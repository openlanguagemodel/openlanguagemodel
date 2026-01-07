import torch
from olm.nn.structure import Block
from olm.nn.embeddings import Embedding
from .transformer_block import TransformerBlock
from .output_head import OutputHead
from olm.nn.structure.combinators import Repeat

class LM(Block):
    """
    A simple Language Model (LM) architecture.

    This model consists of an embedding layer, a stack of Transformer blocks,
    and a final output projection to the vocabulary size. It is designed for
    causal language modeling (next-token prediction).

    Structure:
        Input IDs -> Embedding -> [TransformerBlock] x N -> OutputHead -> Logits

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embeddings and hidden states.
        num_heads (int): Number of attention heads in Transformer blocks.
        num_layers (int): Number of Transformer blocks.
        max_seq_len (int): Maximum sequence length for the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal masking. Defaults to True.
        ff_multiplier (float, optional): Multiplier for FFN hidden dimension. Defaults to 2.5.

    Attributes:
        layers (nn.ModuleList): The sequence of layers in the model.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = True,
        ff_multiplier: float = 2.5,
    ):
        super().__init__([
            # Embedding
            Embedding(vocab_size, embed_dim),

            # Stack of transformer blocks
            Repeat(
                lambda: TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    causal=causal,
                    ff_multiplier=ff_multiplier,
                ),
                num_layers,
            ),

            # Final projection to logits
            OutputHead(embed_dim, vocab_size),
        ])
