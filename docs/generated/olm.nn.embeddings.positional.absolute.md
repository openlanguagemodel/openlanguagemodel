# olm.nn.embeddings.positional.absolute

### Classes

| [`AbsolutePositionalEmbedding`](#olm.nn.embeddings.positional.absolute.AbsolutePositionalEmbedding)(\*args, \*\*kwargs)   | Absolute (Learned) Positional Embedding.   |
|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|

### *class* olm.nn.embeddings.positional.absolute.AbsolutePositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.embeddings.positional.absolute.PositionalEmbeddingBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all positional embedding implementations.

Positional embeddings add information about token positions in a sequence
to help the model understand order and relative positions. Different positional
embedding strategies have different properties:

- Learned (Absolute): Simple, effective, but limited to max_seq_len
- Sinusoidal: Deterministic, can extrapolate to longer sequences
- RoPE: Applied to Q/K directly, enables relative position modeling
- ALiBi: Adds bias to attention scores, excellent extrapolation

All positional embedding implementations should inherit from this base class
and implement the forward method.

#### extra_repr() → str

String representation of the module for debugging.

Override this in subclasses to provide useful information.

#### *abstractmethod* forward(\*args, \*\*kwargs) → torch.Tensor

Apply positional information to input tensor(s).

The signature and behavior of this method varies by implementation:
- Some add to embeddings (Absolute, Sinusoidal)
- Some rotate representations (RoPE)
- Some return bias to add to attention scores (ALiBi)

* **Returns:**
  Transformed tensor(s) with positional information applied
