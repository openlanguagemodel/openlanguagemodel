from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class Gemma2Block(Block):
    """
    A single Transformer block for Gemma 2.
    
    Implements the "Sandwich" Normalization pattern:
    Norm -> Attn -> Norm -> Residual
    Norm -> MLP  -> Norm -> Residual
    """
    def __init__(self, embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, head_dim: int):
        attn_sublayer = Block([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-6),
                GroupedQueryAttention(
                    embed_dim, num_heads, num_kv_heads, max_seq_len, 
                    head_dim=head_dim, dropout=dropout, rope_theta=rope_theta, use_bias=False,
                    use_qk_norm=True, # Gemma 2 uses QK-Norm
                    rms_norm_eps=1e-6
                )
            ])),
            RMSNorm(embed_dim, eps=1e-6)
        ])
        
        mlp_sublayer = Block([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-6),
                GeGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ])),
            RMSNorm(embed_dim, eps=1e-6)
        ])
        
        super().__init__([attn_sublayer, mlp_sublayer])

class Gemma2Model(Block):
    """
    Base class for Gemma 2 models.
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0):
        super().__init__([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: Gemma2Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, head_dim
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-6),
            Linear(embed_dim, vocab_size, bias=False) # torch.nn.Linear can also be used
        ])
        
        # Tie weights: Output head linear = Embedding
        # Gemma 2 ties weights.
        # self.blocks[0] is Embedding wrapper
        # self.blocks[3] is Linear head
        self.blocks[3].weight = self.blocks[0].embedding.weight

class Gemma2_27B(Gemma2Model):
    """Gemma 2 27B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=4608,
            intermediate_size=36864,
            num_layers=46,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128, 
            max_seq_len=8192,
            rope_theta=10000.0
        )

class Gemma2_9B(Gemma2Model):
    """Gemma 2 9B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=3584,
            intermediate_size=14336,
            num_layers=42,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0
        )

class Gemma2_2B(Gemma2Model):
    """Gemma 2 2B Model."""
    def __init__(self):
        super().__init__(
            vocab_size=256000,
            embed_dim=2304,
            intermediate_size=9216,
            num_layers=26,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            max_seq_len=8192,
            rope_theta=10000.0
        )
