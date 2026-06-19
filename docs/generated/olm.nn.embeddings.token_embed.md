# `olm.nn.embeddings.token_embed`

## Classes

### `Embedding(vocab_size: int, embedding_dim: int)`

Token Embedding layer.

Wraps standard PyTorch embedding with a clean interface.
Maps integer indices to dense vectors.

Args:
    vocab_size (int): Size of the vocabulary.
    embedding_dim (int): Dimensionality of the word embeddings.

Attributes:
    embedding (nn.Embedding): The underlying PyTorch embedding layer.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass of the Embedding layer.
