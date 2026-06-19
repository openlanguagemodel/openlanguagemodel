# `olm.nn.attention.base`

Source: [`src/olm/nn/attention/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L1)

## Classes

### `AttentionBase(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True)`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/attention/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L8)

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

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/base.py:57`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L57)

Computes the attention scores and output.

Args:
    q (torch.Tensor): Query tensor [batch, heads, seq, head_dim].
    k (torch.Tensor): Key tensor [batch, heads, seq, head_dim].
    v (torch.Tensor): Value tensor [batch, heads, seq, head_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: The attention output [batch, heads, seq, head_dim].

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/base.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L73)

Standard forward pass for attention layers.

Projects input to Q, K, V, calls `compute_attention`, and projects output.

Args:
    x (torch.Tensor): Input tensor [batch, seq, embed_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: Output tensor [batch, seq, embed_dim].

### `AttentionwithRoPEBase(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, bias: bool = True, rope_theta: float = 10000.0)`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/attention/base.py:95`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L95)

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

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/base.py:148`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L148)

Computes the attention scores and output.

Args:
    q (torch.Tensor): Query tensor [batch, heads, seq, head_dim].
    k (torch.Tensor): Key tensor [batch, heads, seq, head_dim].
    v (torch.Tensor): Value tensor [batch, heads, seq, head_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: The attention output [batch, heads, seq, head_dim].

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/base.py:164`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L164)

Standard forward pass for attention layers.

Projects input to Q, K, V, calls `compute_attention`, and projects output.

Args:
    x (torch.Tensor): Input tensor [batch, seq, embed_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: Output tensor [batch, seq, embed_dim].
