# `olm.nn.embeddings.token_embed`

Source: [`src/olm/nn/embeddings/token_embed.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/token_embed.py#L1)

## Classes

### `Embedding(vocab_size: int, embedding_dim: int)`

**Bases:** `Module`

Source: [`src/olm/nn/embeddings/token_embed.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/token_embed.py#L5)

Token Embedding layer.

Wraps standard PyTorch embedding with a clean interface.
Maps integer indices to dense vectors.

**Parameters**

- `vocab_size` (`int`): Size of the vocabulary.
- `embedding_dim` (`int`): Dimensionality of the word embeddings.

**Attributes**

- `embedding` (`nn.Embedding`): The underlying PyTorch embedding layer.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/token_embed.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/token_embed.py#L30)

Forward pass of the Embedding layer.

**Parameters**

- `x` (`torch.Tensor`): Input tensor of shape (batch_size, seq_len) containing token IDs.

**Returns**

- `torch.Tensor`: Output tensor of shape (batch_size, seq_len, embedding_dim).
