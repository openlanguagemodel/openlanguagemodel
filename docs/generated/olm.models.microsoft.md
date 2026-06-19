# `olm.models.microsoft`

## Classes

### `Phi3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, activation: str = 'swiglu', dropout: float = 0.0, tie_weights: bool = True)`

Base class for Phi 3 models.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Phi-3/Phi-3.5 official checkpoints use
    specialized LongRoPE/scaled RoPE behavior for long contexts, so exact
    long-context behavior may differ from the released Microsoft checkpoints.

### `Phi3_5_Mini()`

Phi-3.5 Mini 3.8B Model.

Uses the public checkpoint dimensions. LongRoPE factors are not represented
by this lightweight preset.

### `Phi3_Small()`

Phi-3 Small 7B Model.

Distinguished by GeGLU activations and the public checkpoint dimensions.
LongRoPE and Phi-3 Small's block-sparse/dense attention schedule are not
represented by this lightweight preset.

### `Phi4Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 250000.0, dropout: float = 0.0, tie_weights: bool = True)`

Base class for Phi 4 models.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`.

### `Phi4_14B()`

Phi-4 14B Model.
