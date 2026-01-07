from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class Qwen2Block(Block):
    """
    A single Transformer block for Qwen 2.
    
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
    def __init__(self, embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, rms_norm_eps: float = 1e-6):
        super().__init__([
            Residual(Block([
                RMSNorm(embed_dim, eps=rms_norm_eps),
                GroupedQueryAttention(
                    embed_dim, num_heads, num_kv_heads, max_seq_len, 
                    dropout=dropout, 
                    rope_theta=rope_theta, 
                    use_bias=False,  # Qwen 2/2.5 uses NO bias in attention
                    use_qk_norm=True, # Qwen 2/2.5 uses QK-Norm
                    rms_norm_eps=rms_norm_eps
                )
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=rms_norm_eps),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])

class Qwen2Model(Block):
    """
    Base class for Qwen 2 / 2.5 models.
    
    Structure:
        Embedding -> [Qwen2Block] x N -> RMSNorm -> Linear Head
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float, tie_weights: bool = False, dropout: float = 0.0, rms_norm_eps: float = 1e-6):
        
        # Core Structure
        layers = [
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: Qwen2Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, rms_norm_eps
            ), num_layers),
            RMSNorm(embed_dim, eps=rms_norm_eps),
            Linear(embed_dim, vocab_size, bias=False) # torch.nn.Linear can also be used
        ]
        
        super().__init__(layers)
        
        # Tie weights logic is post-hoc, but Block init doesn't handle it easily unless we hook it.
        # But we can access the created modules in self.blocks
        if tie_weights:
            # Output head weight (last block) = Embedding weight (first block)
            # Embedding is blocks[0], Linear is blocks[3]
            # blocks[0] is our Wrapper Embedding class, so we access .embedding.weight
            self.blocks[3].weight = self.blocks[0].embedding.weight

# --- Qwen 2.5 Family ---

class Qwen2_5_72B(Qwen2Model):
    """Qwen 2.5 72B Model."""
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
            tie_weights=False,
            rms_norm_eps=1e-5
        )

class Qwen2_5_32B(Qwen2Model):
    """Qwen 2.5 32B Model."""
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

class Qwen2_5_14B(Qwen2Model):
    """Qwen 2.5 14B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=152064,
            embed_dim=5120,
            intermediate_size=13824,
            num_layers=48,
            num_heads=40,
            num_kv_heads=8,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=False
        )

class Qwen2_5_7B(Qwen2Model):
    """Qwen 2.5 7B Model."""
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
        
class Qwen2_5_3B(Qwen2Model):
    """Qwen 2.5 3B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=151936,
            embed_dim=2048,
            intermediate_size=11008, 
            num_layers=36,
            num_heads=16,
            num_kv_heads=2,
            max_seq_len=32768, 
            rope_theta=1000000.0,
            tie_weights=True
        )

class Qwen2_5_1_5B(Qwen2Model):
    """Qwen 2.5 1.5B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=151936,
            embed_dim=1536,
            intermediate_size=8960,
            num_layers=28,
            num_heads=12,
            num_kv_heads=2,
            max_seq_len=131072,
            rope_theta=1000000.0,
            tie_weights=True
        )

class Qwen2_5_0_5B(Qwen2Model):
    """Qwen 2.5 0.5B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=151936,
            embed_dim=896,
            intermediate_size=4864,
            num_layers=24,
            num_heads=14,
            num_kv_heads=2,
            max_seq_len=32768,
            rope_theta=1000000.0,
            tie_weights=True
        )
