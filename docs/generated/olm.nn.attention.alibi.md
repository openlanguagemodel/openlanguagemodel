# `olm.nn.attention.alibi`

## Classes

### `MultiHeadAttentionwithALiBi(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False, causal: bool = True, max_seq_len: int = 2048)`

Multi-Head Attention with ALiBi (Attention with Linear Biases).

ALiBi adds a static, non-learned bias to attention scores based on the distance between
query and key positions. This allows the model to extrapolate to longer sequence lengths
than seen during training.

Args:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    bias (bool, optional): Whether to use bias in linear projections. Defaults to False.
    causal (bool, optional): Whether to apply causal masking logic. Defaults to True.
    max_seq_len (int, optional): Max sequence length for precomputing ALiBi bias. Defaults to 2048.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes attention scores with ALiBi bias.
