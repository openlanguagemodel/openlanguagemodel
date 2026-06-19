# `olm.models.meta`

## Classes

### `Llama2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, tie_weights: bool = True)`

Base class for Llama 2 models.

Structure:
    Embedding -> [Llama2Block] x N -> RMSNorm -> Linear Head

### `Llama2_13B()`

Llama 2 13B (MHA).

### `Llama2_70B()`

Llama 2 70B (GQA).

### `Llama2_7B()`

Llama 2 7B (MHA).

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
