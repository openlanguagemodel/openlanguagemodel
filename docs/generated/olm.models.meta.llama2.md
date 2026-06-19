# `olm.models.meta.llama2`

## Classes

### `Llama2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

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

### `Llama2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0)`

Base class for Llama 2 models.

Structure:
    Embedding -> [Llama2Block] x N -> RMSNorm -> Linear Head

### `Llama2_13B()`

Llama 2 13B (MHA).

### `Llama2_70B()`

Llama 2 70B (GQA).

### `Llama2_7B()`

Llama 2 7B (MHA).
