"""Thin wrappers around torch.nn modules.

Example::
    Block([
        Embedding(vocab_size, embed_dim),
        AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
    ])
"""

import torch.nn as nn

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return super().forward(x)
