# `olm.models.facebook`

## Classes

### `OPT125M()`

OPT 125M Model Definition.

### `OPTModel(vocab_size, embed_dim, intermediate_size, num_layers, num_heads, dropout=0.1)`

OPT Model Definition.

Implements a decoder-only Transformer with specific OPT optimizations:
- Pre-normalization with LayerNorm
- Multi-Head Attention with Causal Masking
- ReLU activation in Feed-Forward Networks

Args:
    vocab_size (int): Vocabulary size.
    embed_dim (int): Embedding dimension.
    intermediate_size (int): FFN dimension.
    num_layers (int): Number of layers.
    num_heads (int): Number of heads.
    dropout (float, optional): Dropout probability. Defaults to 0.1.
