from olm.nn.structure.block import Block
from olm.nn.norms import LayerNorm
from torch import nn


class OutputHead(Block):
    def __init__(self, embed_dim, vocab_size, bias=False):
        super().__init__([
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size, bias=bias),
        ])
