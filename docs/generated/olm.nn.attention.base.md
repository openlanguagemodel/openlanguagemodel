# `olm.nn.attention.base`

## Classes

### `AttentionBase(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True)`

Abstract base class for attention mechanisms.

Provides the common structure for attention layers, including QKV projections
and output projection. Subclasses must implement the specific attention logic
in `compute_attention`.

Attributes:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    head_dim (int): Dimension of each attention head.
    scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
    dropout (nn.Dropout): Dropout layer applied to attention weights.
    q_proj (Linear): Linear projection for Query.
    k_proj (Linear): Linear projection for Key.
    v_proj (Linear): Linear projection for Value.
    out_proj (Linear): Linear projection for Output.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the attention scores and output.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Standard forward pass for attention layers.

### `AttentionwithRoPEBase(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, bias: bool = True, rope_theta: float = 10000.0)`

Abstract base class for attention mechanisms with Rotary Positional Embedding.

Provides the common structure for attention layers, including QKV projections
and output projection. Subclasses must implement the specific attention logic
in `compute_attention`.

Attributes:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    head_dim (int): Dimension of each attention head.
    scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
    dropout (nn.Dropout): Dropout layer applied to attention weights.
    q_proj (Linear): Linear projection for Query.
    k_proj (Linear): Linear projection for Key.
    v_proj (Linear): Linear projection for Value.
    out_proj (Linear): Linear projection for Output.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the attention scores and output.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Standard forward pass for attention layers.
