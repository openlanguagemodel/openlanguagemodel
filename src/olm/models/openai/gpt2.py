from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import FlashAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm
from olm.nn.embeddings import Embedding, AbsolutePositionalEmbedding
from olm.nn.blocks import OutputHead

class GPT2Block(Block):
    """
    A single Transformer block for GPT-2.
    
    Structure:
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
        
    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__([
            Residual(Block([
                LayerNorm(embed_dim),
                FlashAttention(embed_dim, num_heads, dropout=dropout, causal=True)
            ])),
            Residual(Block([
                LayerNorm(embed_dim),
                ClassicFFN(embed_dim, dropout=dropout)
            ]))
        ])

class GPT2Model(Block):
    """
    Base class for GPT-2 models.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        # GPT-2 Logic:
        # Embedding, PositionalEmbedding, Stack of Blocks, OutputHead(LN + Linear)
        
        super().__init__([
            Block([
                Embedding(vocab_size, embed_dim),
                AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout)
            ]),
            Repeat(lambda: GPT2Block(embed_dim, num_heads, dropout), num_layers),
            OutputHead(embed_dim, vocab_size)
        ])
        
        # Tie weights: Output head linear = Embedding
        # Structure:
        #   self.blocks[0] -> Block([Embedding, PosEmbedding])
        #     -> blocks[0] is Embedding wrapper
        #   self.blocks[2] -> OutputHead([LayerNorm, Linear])
        #     -> blocks[1] is Linear
        
        # Accessing the Embedding wrapper's internal embedding module weight
        self.blocks[2].blocks[1].weight = self.blocks[0].blocks[0].embedding.weight

class GPT2(GPT2Model):
    """GPT-2 Small (124M)."""
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            max_seq_len=1024
        )

class GPT2Medium(GPT2Model):
    """GPT-2 Medium (355M)."""
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            embed_dim=1024,
            num_layers=24,
            num_heads=16,
            max_seq_len=1024
        )

class GPT2Large(GPT2Model):
    """GPT-2 Large (774M)."""
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            embed_dim=1280,
            num_layers=36,
            num_heads=20,
            max_seq_len=1024
        )

class GPT2XL(GPT2Model):
    """GPT-2 XL (1.5B)."""
    def __init__(self):
        super().__init__(
            vocab_size=50257,
            embed_dim=1600,
            num_layers=48,
            num_heads=25,
            max_seq_len=1024
        )
