# olm.nn.embeddings

### *class* olm.nn.embeddings.AbsolutePositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`PositionalEmbeddingBase`](olm.nn.embeddings.positional.base.md#olm.nn.embeddings.positional.base.PositionalEmbeddingBase)

Absolute (Learned) Positional Embedding.

This is the standard positional embedding used in the original Transformer paper
and models like GPT-2. It learns a separate embedding vector for each position
in the sequence, up to a maximum sequence length.

These embeddings are typically added to token embeddings before passing through
the transformer blocks.

#### forward(x: torch.Tensor, seq_positions: torch.LongTensor | None = None) → torch.Tensor

Apply absolute positional embedding to input tensor x.

* **Parameters:**
  * **x** – shape (batch_size, seq_len, embed_dim) - token embeddings
  * **seq_positions** – optional tensor of shape (batch_size, seq_len) with position indices.
    If None, assumes positions are 0..seq_len-1 for each batch.
* **Returns:**
  Tensor of same shape as x, with positional embeddings added.

### *class* olm.nn.embeddings.Embedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### Modules

| [`positional`](olm.nn.embeddings.positional.md#module-olm.nn.embeddings.positional)    |    |
|----------------------------------------------------------------------------------------|----|
| [`token_embed`](olm.nn.embeddings.token_embed.md#module-olm.nn.embeddings.token_embed) |    |
