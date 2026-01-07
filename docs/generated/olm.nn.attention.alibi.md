# olm.nn.attention.alibi

### Classes

| [`MultiHeadAttentionwithALiBi`](#olm.nn.attention.alibi.MultiHeadAttentionwithALiBi)(\*args, \*\*kwargs)   | Multi-Head Attention with ALiBi (Attention with Linear Biases).   |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|

### *class* olm.nn.attention.alibi.ALiBiPositionalBias(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.alibi.AttentionBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for attention mechanisms.

Provides the common structure for attention layers, including QKV projections
and output projection. Subclasses must implement the specific attention logic
in compute_attention.

#### embed_dim

Total dimension of the model.

* **Type:**
  int

#### num_heads

Number of parallel attention heads.

* **Type:**
  int

#### head_dim

Dimension of each attention head.

* **Type:**
  int

#### scale

Scaling factor for dot products (1 / sqrt(head_dim)).

* **Type:**
  float

#### dropout

Dropout layer applied to attention weights.

* **Type:**
  nn.Dropout

#### q_proj

Linear projection for Query.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### k_proj

Linear projection for Key.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### v_proj

Linear projection for Value.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### out_proj

Linear projection for Output.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### *abstractmethod* compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes the attention scores and output.

* **Parameters:**
  * **q** (*torch.Tensor*) – Query tensor [batch, heads, seq, head_dim].
  * **k** (*torch.Tensor*) – Key tensor [batch, heads, seq, head_dim].
  * **v** (*torch.Tensor*) – Value tensor [batch, heads, seq, head_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask. Defaults to None.
* **Returns:**
  The attention output [batch, heads, seq, head_dim].
* **Return type:**
  torch.Tensor

#### forward(x: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Standard forward pass for attention layers.

Projects input to Q, K, V, calls compute_attention, and projects output.

* **Parameters:**
  * **x** (*torch.Tensor*) – Input tensor [batch, seq, embed_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask. Defaults to None.
* **Returns:**
  Output tensor [batch, seq, embed_dim].
* **Return type:**
  torch.Tensor

### *class* olm.nn.attention.alibi.MultiHeadAttentionwithALiBi(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionBase)

Multi-Head Attention with ALiBi (Attention with Linear Biases).

ALiBi adds a static, non-learned bias to attention scores based on the distance between
query and key positions. This allows the model to extrapolate to longer sequence lengths
than seen during training.

* **Parameters:**
  * **embed_dim** (*int*) – Total dimension of the model.
  * **num_heads** (*int*) – Number of parallel attention heads.
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **bias** (*bool* *,* *optional*) – Whether to use bias in linear projections. Defaults to False.
  * **causal** (*bool* *,* *optional*) – Whether to apply causal masking logic. Defaults to True.
  * **max_seq_len** (*int* *,* *optional*) – Max sequence length for precomputing ALiBi bias. Defaults to 2048.

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes attention scores with ALiBi bias.
