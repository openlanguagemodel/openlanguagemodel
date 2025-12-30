from olm.nn.structure.block import Block
from olm.nn.norms import LayerNorm
from torch import nn
import torch


class OutputHead(Block):
    """
    Final output projection layer for the Language Model.

    Consists of a LayerNorm followed by a Linear projection to the vocabulary size.
    Typical structure: LayerNorm -> Linear(vocab_size).

    Args:
        embed_dim (int): The dimension of the embedding space.
        vocab_size (int): The size of the vocabulary.
        bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.

    Attributes:
        layers (nn.ModuleList): The normalization and linear layers.
    """
    def __init__(self, embed_dim: int, vocab_size: int, bias: bool = False):
        super().__init__([
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size, bias=bias),
        ])
