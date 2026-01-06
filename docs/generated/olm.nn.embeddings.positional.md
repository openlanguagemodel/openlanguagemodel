# olm.nn.embeddings.positional

### *class* olm.nn.embeddings.positional.ALiBiPositionalBias(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`PositionalEmbeddingBase`](olm.nn.embeddings.positional.base.md#olm.nn.embeddings.positional.base.PositionalEmbeddingBase)

Attention with Linear Biases (ALiBi) as described in
“Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”
(arXiv 2108.12409).

Instead of adding positional information to embeddings, ALiBi adds a bias to attention scores
that is proportional to the distance between query and key positions. This allows the model
to extrapolate to longer sequences than seen during training.

The bias is computed as: `bias[i,j] = -m * |i - j|`
where m is a head-specific slope.

#### forward(seq_len_q: int, seq_len_k: int, device: torch.device | None = None) → torch.Tensor

Get ALiBi bias for the given query and key sequence lengths.

* **Parameters:**
  * **seq_len_q** – length of query sequence
  * **seq_len_k** – length of key sequence (usually same as seq_len_q)
  * **device** – device to place the bias tensor on
* **Returns:**
  Bias tensor of shape (1, num_heads, seq_len_q, seq_len_k)
  This should be added to attention scores before softmax.

### *class* olm.nn.embeddings.positional.AbsolutePositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.embeddings.positional.PartialRotaryPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`PositionalEmbeddingBase`](olm.nn.embeddings.positional.base.md#olm.nn.embeddings.positional.base.PositionalEmbeddingBase)

Partial Rotary Positional Embedding (LLaMA-style RoPE).

Only applies rotary embeddings to a fraction of the head dimensions,
leaving the remaining dimensions unchanged. This is the approach used
in models like LLaMA, where typically 25-50% of dimensions are rotated.

For example, with head_dim=128 and rotary_percentage=0.5, only the first
64 dimensions are rotated, while the last 64 dimensions pass through unchanged.

#### forward(x: torch.Tensor, seq_positions: torch.LongTensor | None = None) → torch.Tensor

Apply partial rotary positional embedding to input tensor x.

* **Parameters:**
  * **x** – shape (batch_size, seq_len, num_heads, head_dim)
  * **seq_positions** – optional tensor of shape (batch_size, seq_len) with position indices.
    If None, assumes positions are 0..seq_len-1 for each batch.
* **Returns:**
  Tensor of same shape as x, with partial RoPE applied.

### *class* olm.nn.embeddings.positional.PositionalEmbeddingBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.embeddings.positional.RotaryPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`PositionalEmbeddingBase`](olm.nn.embeddings.positional.base.md#olm.nn.embeddings.positional.base.PositionalEmbeddingBase)

Rotary Positional Embedding (RoPE) as described in
“RoFormer: Enhanced Transformer with Rotary Position Embedding” (arXiv 2104.09864).

This module precomputes sin/cos rotation frequencies for a given head‐dim, and then applies to
query/key representations via interleaving real/imag parts (or equivalently pairs of dims).

#### forward(x: torch.Tensor, seq_positions: torch.LongTensor | None = None) → torch.Tensor

Apply rotary positional embedding to input tensor x.

* **Parameters:**
  * **x** – shape (batch_size, seq_len, num_heads, head_dim)
  * **seq_positions** – optional tensor of shape (batch_size, seq_len) with position indices.
    If None, assumes positions are 0..seq_len-1 for each batch.
* **Returns:**
  Tensor of same shape as x, with RoPE applied.

### *class* olm.nn.embeddings.positional.SinusoidalPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### Modules

| [`absolute`](olm.nn.embeddings.positional.absolute.md#module-olm.nn.embeddings.positional.absolute)       |    |
|-----------------------------------------------------------------------------------------------------------|----|
| [`alibi`](olm.nn.embeddings.positional.alibi.md#module-olm.nn.embeddings.positional.alibi)                |    |
| [`base`](olm.nn.embeddings.positional.base.md#module-olm.nn.embeddings.positional.base)                   |    |
| [`rope`](olm.nn.embeddings.positional.rope.md#module-olm.nn.embeddings.positional.rope)                   |    |
| [`sinusoidal`](olm.nn.embeddings.positional.sinusoidal.md#module-olm.nn.embeddings.positional.sinusoidal) |    |
