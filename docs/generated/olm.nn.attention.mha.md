# `olm.nn.attention.mha`

## Classes

### `MultiHeadAttention(embed_dims: int, num_heads: int, dropout: float = 0.0, causal: bool = False)`

Implements Multi-Head Attention (MHA) as described in "Attention Is All You Need".

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Supports causal masking for autoregressive models.

Args:
    embed_dims (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
    causal (bool, optional): If True, applies a causal mask. Defaults to False.

Attributes:
    scale (float): Scaling factor (1 / sqrt(head_dim)).
    causal (bool): Whether to apply a causal mask.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the scaled dot-product attention.

### `MultiHeadAttentionwithRoPE(embed_dims: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True, rope_theta: float = 10000.0)`

Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Uses RoPE for positional information.

Args:
    embed_dims (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    max_seq_len (int): Maximum sequence length.
    dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
    causal (bool, optional): If True, applies a causal mask. Defaults to False.

Attributes:
    scale (float): Scaling factor (1 / sqrt(head_dim)).
    causal (bool): Whether to apply a causal mask.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the scaled dot-product attention suited for RoPE.
