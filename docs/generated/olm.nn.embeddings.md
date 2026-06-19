# `olm.nn.embeddings`

## Classes

### `AbsolutePositionalEmbedding(max_seq_len: int, embed_dim: int, dropout: float = 0.0)`

Absolute (Learned) Positional Embedding.

This is the standard positional embedding used in the original Transformer paper
and models like GPT-2. It learns a separate embedding vector for each position
in the sequence, up to a maximum sequence length.

These embeddings are typically added to token embeddings before passing through
the transformer blocks.

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply absolute positional embedding to input tensor x.

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
