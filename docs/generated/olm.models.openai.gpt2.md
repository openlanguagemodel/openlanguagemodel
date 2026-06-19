# `olm.models.openai.gpt2`

## Classes

### `GPT2()`

GPT-2 Small (124M).

### `GPT2Block(embed_dim: int, num_heads: int, dropout: float = 0.1)`

A single Transformer block for GPT-2.

Structure:
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))

Args:
    embed_dim (int): Model dimension.
    num_heads (int): Number of attention heads.
    dropout (float): Dropout probability.

### `GPT2Large()`

GPT-2 Large (774M).

### `GPT2Medium()`

GPT-2 Medium (355M).

### `GPT2Model(vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1)`

Base class for GPT-2 models.

### `GPT2XL()`

GPT-2 XL (1.5B).
