# `olm.models.alibaba.qwen2`

## Classes

### `Qwen2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, rms_norm_eps: float = 1e-06)`

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

### `Qwen2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float, tie_weights: bool = False, dropout: float = 0.0, rms_norm_eps: float = 1e-06)`

Base class for Qwen 2 / 2.5 models.

Structure:
    Embedding -> [Qwen2Block] x N -> RMSNorm -> Linear Head

### `Qwen2_5_0_5B()`

Qwen 2.5 0.5B Model.

### `Qwen2_5_14B()`

Qwen 2.5 14B Model.

### `Qwen2_5_1_5B()`

Qwen 2.5 1.5B Model.

### `Qwen2_5_32B()`

Qwen 2.5 32B Model.

### `Qwen2_5_3B()`

Qwen 2.5 3B Model.

### `Qwen2_5_72B()`

Qwen 2.5 72B Model.

### `Qwen2_5_7B()`

Qwen 2.5 7B Model.
