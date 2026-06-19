# `olm.nn.attention.mha`

Source: [`src/olm/nn/attention/mha.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/mha.py#L1)

## Classes

### `MultiHeadAttention(embed_dims: int, num_heads: int, dropout: float = 0.0, causal: bool = False)`

**Bases:** `olm.nn.attention.base.AttentionBase`

Source: [`src/olm/nn/attention/mha.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/mha.py#L8)

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

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor` (inherited from `AttentionBase`)

Source: [`src/olm/nn/attention/base.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L73)

Standard forward pass for attention layers.

Projects input to Q, K, V, calls `compute_attention`, and projects output.

Args:
    x (torch.Tensor): Input tensor [batch, seq, embed_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: Output tensor [batch, seq, embed_dim].

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/mha.py:29`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/mha.py#L29)

Computes the scaled dot-product attention.

Args:
    q (torch.Tensor): Query tensor of shape [batch, heads, seq, head_dim].
    k (torch.Tensor): Key tensor of shape [batch, heads, seq, head_dim].
    v (torch.Tensor): Value tensor of shape [batch, heads, seq, head_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: The result of the attention mechanism applied to v.

### `MultiHeadAttentionwithRoPE(embed_dims: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True, rope_theta: float = 10000.0)`

**Bases:** `olm.nn.attention.base.AttentionwithRoPEBase`

Source: [`src/olm/nn/attention/mha.py:58`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/mha.py#L58)

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

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor` (inherited from `AttentionwithRoPEBase`)

Source: [`src/olm/nn/attention/base.py:164`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/base.py#L164)

Standard forward pass for attention layers.

Projects input to Q, K, V, calls `compute_attention`, and projects output.

Args:
    x (torch.Tensor): Input tensor [batch, seq, embed_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: Output tensor [batch, seq, embed_dim].

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/mha.py:80`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/mha.py#L80)

Computes the scaled dot-product attention suited for RoPE.

Args:
    q (torch.Tensor): Query tensor of shape [batch, heads, seq, head_dim].
    k (torch.Tensor): Key tensor of shape [batch, heads, seq, head_dim].
    v (torch.Tensor): Value tensor of shape [batch, heads, seq, head_dim].
    mask (torch.Tensor, optional): Attention mask. Defaults to None.

Returns:
    torch.Tensor: The result of the attention mechanism applied to v.
