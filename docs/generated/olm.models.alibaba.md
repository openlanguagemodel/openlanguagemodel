# `olm.models.alibaba`

## Classes

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
