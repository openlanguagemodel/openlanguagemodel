# `olm.nn.embeddings.positional.sinusoidal`

Source: [`src/olm/nn/embeddings/positional/sinusoidal.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/sinusoidal.py#L1)

## Classes

### `SinusoidalPositionalEmbedding(embed_dim: int, max_seq_len: int = 5000, base: int = 10000, dropout: float = 0.0)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/sinusoidal.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/sinusoidal.py#L9)

Sinusoidal Positional Embedding as described in
"Attention Is All You Need" (Vaswani et al., 2017).

Uses fixed sine and cosine functions of different frequencies to encode positions.
Unlike learned embeddings, these are deterministic and can extrapolate to longer
sequences than seen during training.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/sinusoidal.py:80`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/sinusoidal.py#L80)

Apply sinusoidal positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, embed_dim) - token embeddings
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
        If None, assumes positions are 0..seq_len-1 for each batch.

Returns:
    Tensor of same shape as x, with positional embeddings added.
