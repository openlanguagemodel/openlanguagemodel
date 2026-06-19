# `olm.models.microsoft.phi3`

## Classes

### `Phi3Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, activation: str = 'swiglu')`

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

### `Phi3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, activation: str = 'swiglu', dropout: float = 0.0, tie_weights: bool = False)`

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
