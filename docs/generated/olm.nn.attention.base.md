# olm.nn.attention.base

### Classes

| [`AttentionBase`](#olm.nn.attention.base.AttentionBase)(\*args, \*\*kwargs)                 | Abstract base class for attention mechanisms.                                  |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`AttentionwithRoPEBase`](#olm.nn.attention.base.AttentionwithRoPEBase)(\*args, \*\*kwargs) | Abstract base class for attention mechanisms with Rotary Positional Embedding. |

### *class* olm.nn.attention.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.nn.attention.base.AttentionBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
  [Linear](#olm.nn.attention.base.Linear)

#### k_proj

Linear projection for Key.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

#### v_proj

Linear projection for Value.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

#### out_proj

Linear projection for Output.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

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

### *class* olm.nn.attention.base.AttentionwithRoPEBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
  [Linear](#olm.nn.attention.base.Linear)

#### k_proj

Linear projection for Key.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

#### v_proj

Linear projection for Value.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

#### out_proj

Linear projection for Output.

* **Type:**
  [Linear](#olm.nn.attention.base.Linear)

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

### *class* olm.nn.attention.base.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)

### *class* olm.nn.attention.base.RotaryPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### olm.nn.attention.base.abstractmethod(funcobj)

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
‘super’ call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

> class C(metaclass=ABCMeta):
> : @abstractmethod
>   def my_abstract_method(self, arg1, arg2, argN):
>   <br/>
>   > …
