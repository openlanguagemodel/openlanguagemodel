# `olm.models.facebook.opt`

## Classes

### `OPT125M()`

OPT 125M Model Definition.

### `OPTBlock(embed_dim: int, intermediate_size: int, num_heads: int, dropout: float = 0.1)`

A single Transformer block for the OPT architecture.

Composes a Residual Multi-Head Attention block and a Residual ReLU
Feed-Forward block, both utilizing Pre-LayerNorm.

Structure:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + ReLU(LayerNorm(x))
Args:
    embed_dim (int): The dimension of the model.
    intermediate_size (int): The hidden dimension of the feed-forward network.
    num_heads (int): Number of attention heads.
    dropout (float): Dropout probability.

### `OPTModel(vocab_size, embed_dim, intermediate_size, num_layers, num_heads, dropout=0.1, tie_weights=True)`

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
