# olm.nn.embeddings.positional.alibi

### Classes

| [`ALiBiPositionalBias`](#olm.nn.embeddings.positional.alibi.ALiBiPositionalBias)(\*args, \*\*kwargs)   | Attention with Linear Biases (ALiBi) as described in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (arXiv 2108.12409).   |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### *class* olm.nn.embeddings.positional.alibi.ALiBiPositionalBias(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.embeddings.positional.alibi.PositionalEmbeddingBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
