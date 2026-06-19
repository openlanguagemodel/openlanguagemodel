# `olm.nn.embeddings.positional.absolute`

Source: [`src/olm/nn/embeddings/positional/absolute.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/absolute.py#L1)

## Classes

### `AbsolutePositionalEmbedding(max_seq_len: int, embed_dim: int, dropout: float = 0.0)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/absolute.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/absolute.py#L8)

Absolute (Learned) Positional Embedding.

This is the standard positional embedding used in the original Transformer paper
and models like GPT-2. It learns a separate embedding vector for each position
in the sequence, up to a maximum sequence length.

These embeddings are typically added to token embeddings before passing through
the transformer blocks.

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/absolute.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/absolute.py#L34)

Apply absolute positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, embed_dim) - token embeddings
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
        If None, assumes positions are 0..seq_len-1 for each batch.

Returns:
    Tensor of same shape as x, with positional embeddings added.
