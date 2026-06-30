# `olm.nn.embeddings.positional.alibi`

Source: [`src/olm/nn/embeddings/positional/alibi.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/alibi.py#L1)

## Classes

### `ALiBiPositionalBias(num_heads: int, max_seq_len: int = 2048)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/alibi.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/alibi.py#L9)

Attention with Linear Biases (ALiBi) as described in
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
(arXiv 2108.12409).

Instead of adding positional information to embeddings, ALiBi adds a bias to attention scores
that is proportional to the distance between query and key positions. This allows the model
to extrapolate to longer sequences than seen during training.

The bias is computed as: ``bias[i,j] = -m * |i - j|``
where m is a head-specific slope.

#### Methods

##### `forward(self, seq_len_q: int, seq_len_k: int, device: torch.device | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/alibi.py:85`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/alibi.py#L85)

Get ALiBi bias for the given query and key sequence lengths.

**Parameters**

- `seq_len_q`: length of query sequence
- `seq_len_k`: length of key sequence (usually same as seq_len_q)
- `device`: device to place the bias tensor on

**Returns**

Bias tensor of shape (1, num_heads, seq_len_q, seq_len_k)
This should be added to attention scores before softmax.
