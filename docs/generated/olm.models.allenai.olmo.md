# `olm.models.allenai.olmo`

## Classes

### `OLMoBlock(embed_dim: int, intermediate_size: int, num_heads: int, max_seq_len: int, dropout: float)`

A single Transformer block for the OLMo architecture.

Structure:
    x = x + Attn(LN(x))
    x = x + SwiGLU(LN(x))

Args:
    embed_dim (int): Model dimension.
    intermediate_size (int): FFN hidden dimension.
    num_heads (int): Number of attention heads.
    max_seq_len (int): Max context.
    dropout (float): Dropout probability.

### `OLMoModel(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0, tie_weights: bool = False)`

Base class for the OLMo (Open Language Model) architecture.

### `OLMo_7B()`

OLMo 7B Model.
