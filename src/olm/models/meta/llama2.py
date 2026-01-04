from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import MultiHeadAttentionwithRoPE, GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class Llama2Block(Block):
    """
    A single Transformer block for Llama 2.
    
    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
        
    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads. If == num_heads, uses MHA. If < num_heads, uses GQA.
        max_seq_len (int): Max sequence length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
    """
    def __init__(self, embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float):
        # Determine attention type
        if num_kv_heads == num_heads:
            attn_layer = MultiHeadAttentionwithRoPE(
                embed_dim, num_heads, max_seq_len, dropout=dropout, rope_theta=rope_theta, use_bias=False
            )
        else:
            attn_layer = GroupedQueryAttention(
                embed_dim, num_heads, num_kv_heads, max_seq_len, dropout=dropout, rope_theta=rope_theta, use_bias=False
            )

        super().__init__([
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                attn_layer
            ])),
            Residual(Block([
                RMSNorm(embed_dim, eps=1e-5),
                SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
            ]))
        ])

class Llama2Model(Block):
    """
    Base class for Llama 2 models.
    
    Structure:
        Embedding -> [Llama2Block] x N -> RMSNorm -> Linear Head
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0):
        super().__init__([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: Llama2Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-5),
            Linear(embed_dim, vocab_size, bias=False) # torch.nn.Linear can also be used
        ])

class Llama2_7B(Llama2Model):
    """Llama 2 7B (MHA)."""
    def __init__(self):
        super().__init__(
            vocab_size=32000,
            embed_dim=4096,
            intermediate_size=11008,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            max_seq_len=4096,
            rope_theta=10000.0
        )

class Llama2_13B(Llama2Model):
    """Llama 2 13B (MHA)."""
    def __init__(self):
        super().__init__(
            vocab_size=32000,
            embed_dim=5120,
            intermediate_size=13824,
            num_layers=40,
            num_heads=40,
            num_kv_heads=40,
            max_seq_len=4096,
            rope_theta=10000.0
        )

class Llama2_70B(Llama2Model):
    """Llama 2 70B (GQA)."""
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
