
import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class QwenBlock(nn.Module):
    """
    A single Transformer block for the Qwen architecture.
    
    Distinctive features include the use of Bias in Query/Key/Value projections
    and a tighter RMSNorm epsilon (1e-6).

    Structure:
        x = x + GQA(RMSNorm(x)) [Bias=True]
        x = x + SwiGLU(RMSNorm(x)) [Bias=False]

    Args:
        embed_dim (int): Model wide dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
    """
    def __init__(self, embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta):
        super().__init__()
        self.block = Block([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-6),
                GroupedQueryAttention(
                    embed_dim, 
                    num_heads, 
                    num_kv_heads, 
                    max_seq_len, 
                    dropout=dropout, 
                    rope_theta=rope_theta,
                    use_bias=True
                )
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-6),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])
        
    def forward(self, x):
        return self.block(x)

class QwenModel(nn.Module):
    """
    Base class for the Qwen 2.5 architecture.

    Features:
    - QKV Bias enabled (unlike Llama).
    - Tied embeddings for smaller models (Input Embedding weight == Output Head weight).
    - Large vocabulary (~152k).

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Embedding dimension.
        intermediate_size (int): FFN dimension.
        num_layers (int): Number of layers.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Maximum context.
        rope_theta (float): RoPE frequency base.
        tie_weights (bool, optional): Whether to tie embedding and output weights. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, vocab_size, embed_dim, intermediate_size, num_layers, num_heads, num_kv_heads, max_seq_len, rope_theta, tie_weights=False, dropout=0.0):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, embed_dim)
        
        self.transformer = Repeat(lambda: QwenBlock(
            embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta
        ), num_layers)
        
        self.norm = RMSNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        if tie_weights:
            self.head.weight = self.embedding.weight
            
        self.model = Block([
            self.embedding,
            self.transformer,
            self.norm,
            self.head
        ])
        
    def forward(self, x):
        return self.model(x)

# --- Qwen 2.5 Family ---

class Qwen2_5_72B(QwenModel):
    """Qwen 2.5 72B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=8192,
            intermediate_size=29568,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=False
        )

class Qwen2_5_32B(QwenModel):
    """Qwen 2.5 32B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=5120,
            intermediate_size=27648,
            num_layers=64,
            num_heads=40,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=False
        )

class Qwen2_5_14B(QwenModel):
    """Qwen 2.5 14B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=5120,
            intermediate_size=24576,
            num_layers=48,
            num_heads=40,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=False
        )

class Qwen2_5_7B(QwenModel):
    """Qwen 2.5 7B Implementation."""
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=3584,
            intermediate_size=18944,
            num_layers=28,
            num_heads=28,
            num_kv_heads=4,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=False
        )
        
class Qwen2_5_3B(QwenModel):
    """
    Qwen 2.5 3B Implementation.
    
    Note: intermediate_size set to 12800 per HF config (estimated).
    """
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=2560,
            intermediate_size=12800, 
            num_layers=36,
            num_heads=16,
            num_kv_heads=2,
            max_seq_len=32768, 
            rope_theta=1000000.0,
            tie_weights=False 
        )

class Qwen2_5_1_5B(QwenModel):
    """
    Qwen 2.5 1.5B Implementation.
    
    Notable for using Tied Weights for embedding and output head.
    Intermediate size estimated at 8960.
    """
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=1536,
            intermediate_size=8960,
            num_layers=28,
            num_heads=12,
            num_kv_heads=2,
            max_seq_len=32768,
            rope_theta=1000000.0,
            tie_weights=True
        )

class Qwen2_5_0_5B(QwenModel):
    """
    Qwen 2.5 0.5B Implementation.
    
    Uses Tied Weights. Intermediate size estimated at 4864.
    """
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=896,
            intermediate_size=4864,
            num_layers=24,
            num_heads=14,
            num_kv_heads=2,
            max_seq_len=32768,
            rope_theta=1000000.0,
            tie_weights=True
        )
