# `olm.nn.embeddings`

Source: [`src/olm/nn/embeddings/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/__init__.py#L1)

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

**Parameters**

- `x`: shape (batch_size, seq_len, embed_dim) - token embeddings
- `seq_positions`: optional tensor of shape (batch_size, seq_len) with position indices. If None, assumes positions are 0..seq_len-1 for each batch.

**Returns**

Tensor of same shape as x, with positional embeddings added.

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
