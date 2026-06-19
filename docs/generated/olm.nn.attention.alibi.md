# `olm.nn.attention.alibi`

Source: [`src/olm/nn/attention/alibi.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/alibi.py#L1)

## Classes

### `MultiHeadAttentionwithALiBi(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False, causal: bool = True, max_seq_len: int = 2048)`

**Bases:** `olm.nn.attention.base.AttentionBase`

Source: [`src/olm/nn/attention/alibi.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/alibi.py#L9)

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

Source: [`src/olm/nn/attention/alibi.py:41`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/alibi.py#L41)

Computes attention scores with ALiBi bias.
