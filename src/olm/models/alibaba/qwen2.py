from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import GroupedQueryAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead

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
                    use_bias=False,  # Qwen 2/2.5 uses NO bias on the output projection
                    qkv_bias=True  # Qwen 2/2.5 uses bias on Q/K/V projections
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
        Embedding -> [Qwen2Block] x N -> RMSNorm -> tied OutputHead.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
        ``[batch, seq_len, vocab_size]``.
    """
    def __init__(self, vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float, tie_weights: bool = True, dropout: float = 0.0, rms_norm_eps: float = 1e-6):

        # Core Structure
        embedding = Embedding(vocab_size, embed_dim)
        layers = [
            embedding,
            Repeat(lambda: Qwen2Block(
                embed_dim, intermediate_size, num_heads, num_kv_heads, max_seq_len, dropout, rope_theta, rms_norm_eps
            ), num_layers),
            RMSNorm(embed_dim, eps=rms_norm_eps),
            OutputHead(
                embed_dim,
                vocab_size,
                tied_embedding=embedding,
                tie_weights=tie_weights,
                use_norm=False,
            )
        ]

        super().__init__(layers)

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
            rms_norm_eps=1e-5
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
            rms_norm_eps=1e-5
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
        )
