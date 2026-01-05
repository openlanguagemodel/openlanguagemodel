import torch.nn as nn

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return super().forward(x)
    
Block([
    Block([
        Embedding(vocab_size, embed_dim),
        AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout)
    ]),
    Repeat(lambda: GPT2Block(embed_dim, num_heads, dropout), num_layers),
    OutputHead(embed_dim, vocab_size)
])