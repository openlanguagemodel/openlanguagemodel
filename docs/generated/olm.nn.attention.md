# olm.nn.attention

### *class* olm.nn.attention.AttentionBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.AttentionwithRoPEBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.FlashAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.FlashAttentionwithRoPE(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.GroupedQueryAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.MultiHeadAttention(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.MultiHeadAttentionwithALiBi(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.attention.MultiHeadAttentionwithRoPE(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### Modules

| [`alibi`](olm.nn.attention.alibi.md#module-olm.nn.attention.alibi)                   |    |
|--------------------------------------------------------------------------------------|----|
| [`base`](olm.nn.attention.base.md#module-olm.nn.attention.base)                      |    |
| [`flash`](olm.nn.attention.flash.md#module-olm.nn.attention.flash)                   |    |
| [`gqa`](olm.nn.attention.gqa.md#module-olm.nn.attention.gqa)                         |    |
| [`linear_attn`](olm.nn.attention.linear_attn.md#module-olm.nn.attention.linear_attn) |    |
| [`mha`](olm.nn.attention.mha.md#module-olm.nn.attention.mha)                         |    |
