"""
GPT Model Definition.

This module will contain the implementation of the GPT (Generative Pre-trained Transformer) architecture.
"""

from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import MultiHeadAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.absolute import AbsolutePositionalEmbedding
from olm.nn.blocks.output_head import OutputHead

class GPT2Block(Block):
    """
    A single Transformer block for GPT-2.

    Structure:
        Input -> Residual(LayerNorm -> MHA) -> Residual(LayerNorm -> FFN) -> Output
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__([
            Residual(
                Block([
                    LayerNorm(embed_dim),
                    MultiHeadAttention(embed_dim, num_heads, dropout=dropout, causal=True),
                ])
            ),
            Residual(
                Block([
                    LayerNorm(embed_dim),
                    ClassicFFN(embed_dim, dropout=dropout),
                ])
            )
        ])

class GPT2(Block):
    """
    GPT-2 124M Model Definition.
    
    Structure:
        Input IDs -> Embedding + PositionalEmbedding -> [GPT2Block] x 12 -> LayerNorm -> OutputHead -> Logits
    """
    def __init__(self):
        # GPT-2 124M Hyperparameters
        vocab_size = 50257
        embed_dim = 768
        num_layers = 12
        num_heads = 12
        max_seq_len = 1024
        dropout = 0.1

        super().__init__([
            # 1. Token Embeddings + Positional Embeddings
            Embedding(vocab_size, embed_dim),
            AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
            
            # 2. Transformer Blocks
            Repeat(
                lambda: GPT2Block(embed_dim, num_heads, dropout),
                num_layers
            ),

            # 3. Final LayerNorm (Included in OutputHead)
            
            # 4. Output Head
            OutputHead(embed_dim, vocab_size)
        ])
        
        # Tie weights: OutputHead Linear weight = Embedding weight
        # GPT2.blocks structure:
        # [0]: Embedding
        # [1]: AbsolutePositionalEmbedding
        # [2]: Repeat(GPT2Block)
        # [3]: OutputHead
        
        # OutputHead.blocks structure:
        # [0]: LayerNorm
        # [1]: Linear
        
        self.blocks[3].blocks[1].weight = self.blocks[0].embedding.weight
