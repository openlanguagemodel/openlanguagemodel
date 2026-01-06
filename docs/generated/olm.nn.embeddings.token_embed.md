# olm.nn.embeddings.token_embed

### Classes

| [`Embedding`](#olm.nn.embeddings.token_embed.Embedding)(\*args, \*\*kwargs)   | Token Embedding layer.   |
|-------------------------------------------------------------------------------|--------------------------|

### *class* olm.nn.embeddings.token_embed.Embedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Token Embedding layer.

Wraps standard PyTorch embedding with a clean interface.
Maps integer indices to dense vectors.

* **Parameters:**
  * **vocab_size** (*int*) – Size of the vocabulary.
  * **embedding_dim** (*int*) – Dimensionality of the word embeddings.

#### embedding

The underlying PyTorch embedding layer.

* **Type:**
  nn.Embedding

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of the Embedding layer.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch_size, seq_len) containing token IDs.
* **Returns:**
  Output tensor of shape (batch_size, seq_len, embedding_dim).
* **Return type:**
  torch.Tensor
