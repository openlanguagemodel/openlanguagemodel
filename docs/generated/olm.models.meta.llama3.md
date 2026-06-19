# `olm.models.meta.llama3`

## Classes

### `Llama3Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

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

### `Llama3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 500000.0, dropout: float = 0.0, tie_weights: bool = True)`

Base class for Llama 3, 3.1, and 3.2 models.

Inherits from Block for pure sequential composition.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Llama 3.1/3.2 official checkpoints use
    specialized scaled RoPE behavior for long contexts, so exact long-context
    behavior may differ from the released Meta checkpoints.

Structure:
    Embedding -> [Llama3Block] x N -> RMSNorm -> Linear Head

### `Llama3_1_405B()`

Llama 3.1 405B Model (Flagship).

### `Llama3_1_70B()`

Llama 3.1 70B Model.

### `Llama3_1_8B()`

Llama 3.1 8B Model.

### `Llama3_2_1B()`

Llama 3.2 1B Model (Pruned/Distilled).

### `Llama3_2_3B()`

Llama 3.2 3B Model (Edge-optimized).
