# olm.nn.attention.flash

### Classes

| [`FlashAttention`](#olm.nn.attention.flash.FlashAttention)(\*args, \*\*kwargs)                 | Flash Attention implementation for efficient attention computation.   |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`FlashAttentionwithRoPE`](#olm.nn.attention.flash.FlashAttentionwithRoPE)(\*args, \*\*kwargs) | Flash Attention implementation for efficient attention computation.   |

### *class* olm.nn.attention.flash.AttentionBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.flash.AttentionwithRoPEBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.flash.FlashAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionBase)

Flash Attention implementation for efficient attention computation.

Uses PyTorch’s native scaled_dot_product_attention (which includes Flash Attention 2
optimizations) when available, or falls back to a memory-efficient implementation.

Flash Attention provides:
- O(N) memory complexity instead of O(N²) for sequence length N
- Faster computation through kernel fusion and tiling
- Exact attention (not an approximation)
- Support for causal masking without materializing the full attention matrix

Reference: “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”
(Dao et al., 2022) and “FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning” (Dao, 2023)

* **Parameters:**
  * **embed_dim** – Total dimension of the model
  * **num_heads** – Number of parallel attention heads
  * **dropout** – Dropout probability on attention weights (default: 0.0)
  * **causal** – If True, applies causal masking for autoregressive models (default: False)
  * **use_flash_attn** – Force enable/disable flash attention. If None, auto-detect (default: None)

### Example

```pycon
>>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
>>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
>>> output = attn(x)
>>> output.shape
torch.Size([2, 128, 512])
```

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes attention using Flash Attention when available.

* **Parameters:**
  * **q** – Query tensor [batch, heads, seq, head_dim]
  * **k** – Key tensor [batch, heads, seq, head_dim]
  * **v** – Value tensor [batch, heads, seq, head_dim]
  * **mask** – Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]
* **Returns:**
  Attention output [batch, heads, seq, head_dim]

#### extra_repr() → str

String representation of the module.

#### forward(x: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Forward pass with Flash Attention.

* **Parameters:**
  * **x** – Input tensor [batch, seq_len, embed_dim]
  * **mask** – Optional attention mask
* **Returns:**
  Output tensor [batch, seq_len, embed_dim]

### *class* olm.nn.attention.flash.FlashAttentionwithRoPE(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionwithRoPEBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionwithRoPEBase)

Flash Attention implementation for efficient attention computation.

Uses PyTorch’s native scaled_dot_product_attention (which includes Flash Attention 2
optimizations) when available, or falls back to a memory-efficient implementation.

Flash Attention provides:
- O(N) memory complexity instead of O(N²) for sequence length N
- Faster computation through kernel fusion and tiling
- Exact attention (not an approximation)
- Support for causal masking without materializing the full attention matrix

Reference: “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”
(Dao et al., 2022) and “FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning” (Dao, 2023)

* **Parameters:**
  * **embed_dim** – Total dimension of the model
  * **num_heads** – Number of parallel attention heads
  * **dropout** – Dropout probability on attention weights (default: 0.0)
  * **causal** – If True, applies causal masking for autoregressive models (default: False)
  * **use_flash_attn** – Force enable/disable flash attention. If None, auto-detect (default: None)

### Example

```pycon
>>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
>>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
>>> output = attn(x)
>>> output.shape
torch.Size([2, 128, 512])
```

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes attention using Flash Attention when available.

* **Parameters:**
  * **q** – Query tensor [batch, heads, seq, head_dim]
  * **k** – Key tensor [batch, heads, seq, head_dim]
  * **v** – Value tensor [batch, heads, seq, head_dim]
  * **mask** – Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]
* **Returns:**
  Attention output [batch, heads, seq, head_dim]

#### extra_repr() → str

String representation of the module.

#### forward(x: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Forward pass with Flash Attention and RoPE.

* **Parameters:**
  * **x** – Input tensor [batch, seq_len, embed_dim]
  * **mask** – Optional attention mask
* **Returns:**
  Output tensor [batch, seq_len, embed_dim]

### *class* olm.nn.attention.flash.RotaryPositionalEmbedding(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
