
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class LlamaBlock(nn.Module):
    """
    A single Transformer block for the Llama architecture.

    Composes a Residual Grouped Query Attention (GQA) block and a Residual SwiGLU 
    Feed-Forward block, both utilizing Pre-RMSNorm.

    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        embed_dim (int): The dimension of the model.
        intermediate_size (int): The hidden dimension of the feed-forward network.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of key/value heads for GQA.
        max_seq_len (int): Maximum sequence length for RoPE.
        dropout (float): Dropout probability.
        rope_theta (float): The base frequency for Rotary Positional Embeddings.
    """
    def __init__(self, embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta):
        super().__init__()
        self.block = Block([
            Residual(Block([
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
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])
        
    def forward(self, x):
        return self.block(x)

class LlamaModel(nn.Module):
    """
    Base class for the Llama model architecture.

    Implements a decoder-only Transformer with specific Llama optimizations:
    - Pre-normalization with RMSNorm
    - SwiGLU activation in Feed-Forward Networks
    - Rotary Positional Embeddings (RoPE)
    - Grouped Query Attention (GQA)

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Model wide embedding dimension.
        intermediate_size (int): Intermediate dimension for FFN.
        num_layers (int): Number of Transformer blocks.
        num_heads (int): Number of Query attention heads.
        num_kv_heads (int): Number of Key/Value attention heads.
        max_seq_len (int): Maximum context length.
        rope_theta (float): RoPE base frequency.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, num_kv_heads, max_seq_len, rope_theta, dropout=0.0):
        super().__init__()
        self.model = Block([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: LlamaBlock(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, vocab_size, bias=False)
        ])
        
    def forward(self, x):
        return self.model(x)

# --- Llama 3.1 Family ---

class Llama3_1_405B(LlamaModel):
    """
    Llama 3.1 405B Model.

    The flagship model of the Llama 3.1 family, utilizing massive scale and high 
    frequency RoPE for long-context support (128k).
    """
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=16384,
            intermediate_size=53248,
            num_layers=126,
            num_heads=128,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )

class Llama3_1_70B(LlamaModel):
    """
    Llama 3.1 70B Model.

    A balanced enterprise-grade model retaining 128k context and GQA.
    """
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=8192,
            intermediate_size=28672,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )

class Llama3_1_8B(LlamaModel):
    """
    Llama 3.1 8B Model.

    Standard edge model with 15T token training duration.
    """
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=4096,
            intermediate_size=14336,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )

# --- Llama 3.2 Family ---

class Llama3_2_3B(LlamaModel):
    """
    Llama 3.2 3B Model.

    Optimized for mobile NPUs with a non-power-of-2 context width (3072).
    """
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=3072,
            intermediate_size=8192,
            num_layers=28,
            num_heads=24,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )

class Llama3_2_1B(LlamaModel):
    """
    Llama 3.2 1B Model.

    Pruned and distilled for extreme efficiency on edge devices.
    """
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=2048,
            intermediate_size=8192,
            num_layers=16,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )

# --- Llama 2 Family ---

class Llama2_70B(LlamaModel):
    """
    Llama 2 70B Model.

    Introduced GQA to the Llama 2 line for inference efficiency.
    """
    def __init__(self):
        super().__init__(
            vocab_size=32000,
            embed_dim=8192,
            intermediate_size=28672,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            max_seq_len=4096,
            rope_theta=10000.0
        )

class Llama2_13B(LlamaModel):
    """
    Llama 2 13B Model.

    Uses standard Multi-Head Attention (MHA).
    """
    def __init__(self):
        super().__init__(
            vocab_size=32000,
            embed_dim=5120,
            intermediate_size=13824,
            num_layers=40,
            num_heads=40,
            num_kv_heads=40, # MHA
            max_seq_len=4096,
            rope_theta=10000.0
        )

class Llama2_7B(LlamaModel):
    """
    Llama 2 7B Model.

    The original open-weight fine-tuning standard. Uses MHA.
    """
    def __init__(self):
        super().__init__(
            vocab_size=32000,
            embed_dim=4096,
            intermediate_size=11008,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32, # MHA
            max_seq_len=4096,
            rope_theta=10000.0
        )
