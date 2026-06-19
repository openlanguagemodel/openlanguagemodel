# `olm.models.microsoft.phi4`

## Classes

### `Phi4Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

A single Transformer block for Phi 4.

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

### `Phi4Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 250000.0, dropout: float = 0.0, tie_weights: bool = True)`

Base class for Phi 4 models.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`.

### `Phi4_14B()`

Phi-4 14B Model.
