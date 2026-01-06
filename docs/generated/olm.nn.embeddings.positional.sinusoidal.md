# olm.nn.embeddings.positional.sinusoidal

### Classes

| [`SinusoidalPositionalEmbedding`](#olm.nn.embeddings.positional.sinusoidal.SinusoidalPositionalEmbedding)(\*args, \*\*kwargs)   | Sinusoidal Positional Embedding as described in "Attention Is All You Need" (Vaswani et al., 2017).   |
|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|

### *class* olm.nn.embeddings.positional.sinusoidal.PositionalEmbeddingBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.embeddings.positional.sinusoidal.SinusoidalPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`PositionalEmbeddingBase`](olm.nn.embeddings.positional.base.md#olm.nn.embeddings.positional.base.PositionalEmbeddingBase)

Sinusoidal Positional Embedding as described in
“Attention Is All You Need” (Vaswani et al., 2017).

Uses fixed sine and cosine functions of different frequencies to encode positions.
Unlike learned embeddings, these are deterministic and can extrapolate to longer
sequences than seen during training.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#### forward(x: torch.Tensor, seq_positions: torch.LongTensor | None = None) → torch.Tensor

Apply sinusoidal positional embedding to input tensor x.

* **Parameters:**
  * **x** – shape (batch_size, seq_len, embed_dim) - token embeddings
  * **seq_positions** – optional tensor of shape (batch_size, seq_len) with position indices.
    If None, assumes positions are 0..seq_len-1 for each batch.
* **Returns:**
  Tensor of same shape as x, with positional embeddings added.
