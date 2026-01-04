from olm.nn import Linear
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention, MultiHeadAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.feedforward.geglu_ffn import GeGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding

class Phi3Block(Block):
    """
    A single Transformer block for Phi 3.
    
    Structure:
        x = x + GQA(RMSNorm(x))
        x = x + FFN(RMSNorm(x))  # FFN can be SwiGLU or GeGLU
        
    Args:
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        max_seq_len (int): Max sequence length.
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base.
        activation (str): "swiglu" or "geglu".
    """
    def __init__(self, embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, activation: str = "swiglu"):
        if activation == "swiglu":
            ffn_layer = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
        elif activation == "geglu":
            ffn_layer = GeGLUFFN(embed_dim, hidden_dim=intermediate_size, dropout=dropout, bias=False)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Use MHA if kv_heads == heads (standard for Phi-3.5 Mini)
        # Use GQA if kv_heads < heads (standard for Phi-3 Small)
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
                ffn_layer
            ]))
        ])

class Phi3Model(Block):
    """
    Base class for Phi 3 models.

    > [!NOTE/TODO]
    > **Architectural Limitation - RoPE Scaling**:
    > This implementation uses standard Rotary Positional Embeddings (RoPE) parameterized via `rope_theta`.
    > Phi-3/Phi-3.5 oficially uses a specialized "Scaled RoPE" (LongRoPE/NTK-aware mix) to support massive contexts (128k).
    > While `max_seq_len` is set to 128k, this implementation lacks the exact scaling logic of the official release,
    > meaning long-context performance may differ.
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__([
            Embedding(vocab_size, embed_dim),
            Repeat(lambda: Phi3Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, activation
            ), num_layers),
            RMSNorm(embed_dim, eps=1e-5),
            Linear(embed_dim, vocab_size, bias=False) # torch.nn.Linear can also be used
        ])
        
        # Tie weights: Output head linear = Embedding
        # Phi-3 ties weights.
        # self.blocks[0] is Embedding wrapper
        # self.blocks[3] is Linear head
        self.blocks[3].weight = self.blocks[0].embedding.weight

class Phi3_5_Mini(Phi3Model):
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
        
class Phi3_Small(Phi3Model):
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
