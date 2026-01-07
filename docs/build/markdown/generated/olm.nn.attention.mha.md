# olm.nn.attention.mha

### Classes

| [`MultiHeadAttention`](#olm.nn.attention.mha.MultiHeadAttention)(\*args, \*\*kwargs)                 | Implements Multi-Head Attention (MHA) as described in "Attention Is All You Need".   |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`MultiHeadAttentionwithRoPE`](#olm.nn.attention.mha.MultiHeadAttentionwithRoPE)(\*args, \*\*kwargs) | Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).       |

### *class* olm.nn.attention.mha.AttentionBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.mha.AttentionwithRoPEBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for attention mechanisms with Rotary Positional Embedding.

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

### *class* olm.nn.attention.mha.MultiHeadAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionBase)

Implements Multi-Head Attention (MHA) as described in “Attention Is All You Need”.

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Supports causal masking for autoregressive models.

* **Parameters:**
  * **embed_dims** (*int*) – Total dimension of the model.
  * **num_heads** (*int*) – Number of parallel attention heads.
  * **dropout** (*float* *,* *optional*) – Dropout probability on attention weights. Defaults to 0.0.
  * **causal** (*bool* *,* *optional*) – If True, applies a causal mask. Defaults to False.

#### scale

Scaling factor (1 / sqrt(head_dim)).

* **Type:**
  float

#### causal

Whether to apply a causal mask.

* **Type:**
  bool

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes the scaled dot-product attention.

* **Parameters:**
  * **q** (*torch.Tensor*) – Query tensor of shape [batch, heads, seq, head_dim].
  * **k** (*torch.Tensor*) – Key tensor of shape [batch, heads, seq, head_dim].
  * **v** (*torch.Tensor*) – Value tensor of shape [batch, heads, seq, head_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask. Defaults to None.
* **Returns:**
  The result of the attention mechanism applied to v.
* **Return type:**
  torch.Tensor

### *class* olm.nn.attention.mha.MultiHeadAttentionwithRoPE(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionwithRoPEBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionwithRoPEBase)

Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Uses RoPE for positional information.

* **Parameters:**
  * **embed_dims** (*int*) – Total dimension of the model.
  * **num_heads** (*int*) – Number of parallel attention heads.
  * **max_seq_len** (*int*) – Maximum sequence length.
  * **dropout** (*float* *,* *optional*) – Dropout probability on attention weights. Defaults to 0.0.
  * **causal** (*bool* *,* *optional*) – If True, applies a causal mask. Defaults to False.

#### scale

Scaling factor (1 / sqrt(head_dim)).

* **Type:**
  float

#### causal

Whether to apply a causal mask.

* **Type:**
  bool

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes the scaled dot-product attention suited for RoPE.

* **Parameters:**
  * **q** (*torch.Tensor*) – Query tensor of shape [batch, heads, seq, head_dim].
  * **k** (*torch.Tensor*) – Key tensor of shape [batch, heads, seq, head_dim].
  * **v** (*torch.Tensor*) – Value tensor of shape [batch, heads, seq, head_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask. Defaults to None.
* **Returns:**
  The result of the attention mechanism applied to v.
* **Return type:**
  torch.Tensor
