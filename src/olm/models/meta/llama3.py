from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class Llama3Block(Block):
    """
    A single Transformer block for Llama 3.x architecture.
    
    Similar to Llama 2 but parameterized for Llama 3's high-performance context.
    
    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
        
    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Max sequence length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
    """
    def __init__(self, embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float):
        super().__init__([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                GroupedQueryAttention(
                    embed_dim, num_heads, num_kv_heads, max_seq_len, dropout=dropout, rope_theta=rope_theta, use_bias=False
                )
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])

class Llama3Model(Block):
    """
    Base class for Llama 3, 3.1, and 3.2 models.
    
    Inherits from Block for pure sequential composition.
    
    > [!NOTE/TODO]
    > **Architectural Limitation - RoPE Scaling**:
    > This implementation uses standard Rotary Positional Embeddings (RoPE) parameterized via `rope_theta`.
    > Llama 3.1/3.2 officially uses a specialized "Scaled RoPE" with frequency interpolation/extrapolation 
    > (often specialized low-freq/high-freq handling) to support massive contexts (128k).
    > While `max_seq_len` is set to 128k, this implementation lacks the exact scaling logic of the official release,
    > meaning long-context performance may differ.
    
    Structure:
        Embedding -> [Llama3Block] x N -> RMSNorm -> Linear Head
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 500000.0, dropout: float = 0.0):
        super().__init__([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: Llama3Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-5),
            Linear(embed_dim, vocab_size, bias=False) # torch.nn.Linear can also be used
        ])

# --- Llama 3.1 ---

class Llama3_1_405B(Llama3Model):
    """Llama 3.1 405B Model (Flagship)."""
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

class Llama3_1_70B(Llama3Model):
    """Llama 3.1 70B Model."""
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

class Llama3_1_8B(Llama3Model):
    """Llama 3.1 8B Model."""
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

# --- Llama 3.2 ---

class Llama3_2_3B(Llama3Model):
    """Llama 3.2 3B Model (Edge-optimized)."""
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

class Llama3_2_1B(Llama3Model):
    """Llama 3.2 1B Model (Pruned/Distilled)."""
    def __init__(self):
        super().__init__(
            vocab_size=128256,
            embed_dim=2048,
            intermediate_size=8192,
            num_layers=16,
            num_heads=16, # Head dim 128
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=500000.0
        )
