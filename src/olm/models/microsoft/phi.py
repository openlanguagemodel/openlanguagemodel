
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.feedforward.geglu_ffn import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class PhiBlock(nn.Module):
    """
    A single Transformer block for the Phi architecture.
    
    Supports selectable activation functions (SwiGLU for standard Phi, GeGLU for Small).
    Uses Standard Pre-Norm structure.

    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Max context.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
        activation (str): "swiglu" or "geglu".
    """
    def __init__(self, embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, activation="swiglu"):
        super().__init__()
        
        self.attn = Residual(Block([
            RMSNorm(embed_dim, eps=1e-5),
            GroupedQueryAttention(
                embed_dim, 
                num_heads, 
                num_kv_heads, 
                max_seq_len, 
                dropout=dropout, 
                rope_theta=rope_theta,
                use_bias=False
            )
        ]))
        
        if activation == "swiglu":
            ffn_cls = SwiGLUFFN
        elif activation == "geglu":
            ffn_cls = GeGLUFFN
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        self.mlp = Residual(Block([
            RMSNorm(embed_dim, eps=1e-5),
            ffn_cls(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
        ]))
        
        self.block = Block([self.attn, self.mlp])
        
    def forward(self, x):
        return self.block(x)

class PhiModel(nn.Module):
    """
    Base class for Microsoft Phi models.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Context length.
        rope_theta (float): RoPE base.
        activation (str, optional): FFN activation type. Defaults to "swiglu".
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, num_kv_heads, max_seq_len, rope_theta, activation="swiglu", dropout=0.0):
        super().__init__()
        self.model = Block([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: PhiBlock(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, activation
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, vocab_size, bias=False)
        ])
    
    def forward(self, x):
        return self.model(x)

class Phi4_14B(PhiModel):
    """Phi-4 14B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=100352,
            embed_dim=5120,
            intermediate_size=17920,
            num_layers=40,
            num_heads=40,
            num_kv_heads=10,
            max_seq_len=16384,
            rope_theta=10000.0,
            activation="swiglu"
        )
        
class Phi3_5_Mini(PhiModel):
    """Phi-3.5 Mini 3.8B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=32064,
            embed_dim=3072,
            intermediate_size=8192,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32, # MHA typically
            max_seq_len=128000,
            rope_theta=10000.0,
            activation="swiglu"
        )
        
class Phi3_Small(PhiModel):
    """
    Phi-3 Small 7B Model.

    Distinguished by the use of GeGLU activations.
    """
    def __init__(self):
        super().__init__(
            vocab_size=100352,
            embed_dim=4096,
            intermediate_size=11008,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=128000,
            rope_theta=10000.0,
            activation="geglu"
        )
