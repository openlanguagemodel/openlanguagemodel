# olm.nn.attention.gqa

### Classes

| [`GroupedQueryAttention`](#olm.nn.attention.gqa.GroupedQueryAttention)(\*args, \*\*kwargs)   | Grouped Query Attention (GQA) with Rotary Positional Embeddings.   |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------|

### *class* olm.nn.attention.gqa.GroupedQueryAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Grouped Query Attention (GQA) with Rotary Positional Embeddings.

GQA is a distinct attention mechanism where the number of Key/Value heads is smaller
than the number of Query heads. This reduces memory bandwidth usage during inference
(smaller KV cache) while maintaining performance close to Multi-Head Attention (MHA).

If num_kv_heads == num_heads, this is equivalent to MHA.
If num_kv_heads == 1, this is equivalent to Multi-Query Attention (MQA).

* **Parameters:**
  * **embed_dim** (*int*) – Total dimension of the model.
  * **num_heads** (*int*) – Number of Query heads.
  * **num_kv_heads** (*int*) – Number of Key/Value heads. Must divide num_heads.
  * **max_seq_len** (*int*) – Maximum sequence length for RoPE.
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **rope_theta** (*float* *,* *optional*) – Base frequency for RoPE. Defaults to 10000.0.
  * **use_bias** (*bool* *,* *optional*) – Whether to use bias in linear projections. Defaults to False.

#### forward(x: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Forward pass of Grouped Query Attention.

* **Parameters:**
  * **x** (*torch.Tensor*) – Input tensor of shape [batch, seq_len, embed_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask of shape [batch, 1, seq_len, seq_len]
    or [batch, seq_len, seq_len]. Defaults to None.
* **Returns:**
  Output tensor of shape [batch, seq_len, embed_dim].
* **Return type:**
  torch.Tensor

### *class* olm.nn.attention.gqa.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)

### *class* olm.nn.attention.gqa.RMSNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`NormBase`](olm.nn.norms.base.md#olm.nn.norms.base.NormBase)

RMSNorm (Root Mean Square Layer Normalization) layer.

Implements RMSNorm as described in “Root Mean Square Layer Normalization” ([https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)).
A simplified version of LayerNorm that scales invariance properties.

* **Parameters:**
  * **d_model** (*int*) – The dimension of the model to normalize.
  * **eps** (*float* *,* *optional*) – Small constant for numerical stability. Defaults to 1e-5.
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### weight

Learnable scale parameter.

* **Type:**
  nn.Parameter

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of RMSNorm.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch_size, sequence_length, d_model).
* **Returns:**
  Normalized output tensor of the same shape.
* **Return type:**
  torch.Tensor

### *class* olm.nn.attention.gqa.RotaryPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
