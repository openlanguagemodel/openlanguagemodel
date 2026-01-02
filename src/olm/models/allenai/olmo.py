
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import MultiHeadAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms.layer_norm import LayerNorm
from olm.nn.embeddings import Embedding

class OLMoBlock(nn.Module):
    """
    A single Transformer block for the OLMo architecture.

    Designed for scientific analysis with minimal 'tricks':
    - Non-Affine LayerNorm (no learnable gamma/beta).
    - No Bias terms in any dense projection (Attention or MLP).
    - SwiGLU activation.

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Max context.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim, intermediate_size, num_heads, max_seq_len, dropout):
        super().__init__()
        self.block = Block([
            Residual(Block([
                LayerNorm(embed_dim, elementwise_affine=False),
                MultiHeadAttentionwithRoPE(
                    embed_dim, 
                    num_heads, 
                    max_seq_len, 
                    dropout=dropout,
                    bias=False # No bias for OLMo
                )
            ])),
            Residual(Block([
                LayerNorm(embed_dim, elementwise_affine=False),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])
        
    def forward(self, x):
        return self.block(x)

class OLMoModel(nn.Module):
    """
    Base class for the OLMo (Open Language Model) architecture.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of heads.
        max_seq_len (int, optional): Context length. Defaults to 4096.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, max_seq_len=4096, dropout=0.0):
        super().__init__()
        self.model = Block([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: OLMoBlock(
                embed_dim, intermediate_size, num_heads, max_seq_len, dropout
            ), num_layers),
            LayerNorm(embed_dim, elementwise_affine=False),
            nn.Linear(embed_dim, vocab_size, bias=False)
        ])
        
    def forward(self, x):
        return self.model(x)

class OLMo_7B(OLMoModel):
    """OLMo 7B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=50280,
            embed_dim=4096,
            intermediate_size=11008,
            num_layers=32,
            num_heads=32,
            max_seq_len=4096
        )
