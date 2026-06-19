# `olm.nn.attention`

## Classes

### `AttentionBase(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True)`

Abstract base class for attention mechanisms.

Provides the common structure for attention layers, including QKV projections
and output projection. Subclasses must implement the specific attention logic
in `compute_attention`.

Attributes:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    head_dim (int): Dimension of each attention head.
    scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
    dropout (nn.Dropout): Dropout layer applied to attention weights.
    q_proj (Linear): Linear projection for Query.
    k_proj (Linear): Linear projection for Key.
    v_proj (Linear): Linear projection for Value.
    out_proj (Linear): Linear projection for Output.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the attention scores and output.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Standard forward pass for attention layers.

### `AttentionwithRoPEBase(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, bias: bool = True, rope_theta: float = 10000.0)`

Abstract base class for attention mechanisms with Rotary Positional Embedding.

Provides the common structure for attention layers, including QKV projections
and output projection. Subclasses must implement the specific attention logic
in `compute_attention`.

Attributes:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    head_dim (int): Dimension of each attention head.
    scale (float): Scaling factor for dot products (1 / sqrt(head_dim)).
    dropout (nn.Dropout): Dropout layer applied to attention weights.
    q_proj (Linear): Linear projection for Query.
    k_proj (Linear): Linear projection for Key.
    v_proj (Linear): Linear projection for Value.
    out_proj (Linear): Linear projection for Output.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the attention scores and output.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Standard forward pass for attention layers.

### `FlashAttention(embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool | None = None)`

Flash Attention implementation for efficient attention computation.

Uses PyTorch's native scaled_dot_product_attention (which includes Flash Attention 2
optimizations) when available, or falls back to a memory-efficient implementation.

Flash Attention provides:
- O(N) memory complexity instead of O(N²) for sequence length N
- Faster computation through kernel fusion and tiling
- Exact attention (not an approximation)
- Support for causal masking without materializing the full attention matrix

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
(Dao et al., 2022) and "FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning" (Dao, 2023)

Args:
    embed_dim: Total dimension of the model
    num_heads: Number of parallel attention heads
    dropout: Dropout probability on attention weights (default: 0.0)
    causal: If True, applies causal masking for autoregressive models (default: False)
    use_flash_attn: Force enable/disable flash attention. If None, auto-detect (default: None)

Example:
    >>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
    >>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
    >>> output = attn(x)
    >>> output.shape
    torch.Size([2, 128, 512])

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes attention using Flash Attention when available.
- `extra_repr(self) -> str`
  String representation of the module.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Forward pass with Flash Attention.

### `FlashAttentionwithRoPE(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True, rope_theta: float = 10000.0, use_flash_attn: bool | None = None)`

Flash Attention implementation for efficient attention computation.

Uses PyTorch's native scaled_dot_product_attention (which includes Flash Attention 2
optimizations) when available, or falls back to a memory-efficient implementation.

Flash Attention provides:
- O(N) memory complexity instead of O(N²) for sequence length N
- Faster computation through kernel fusion and tiling
- Exact attention (not an approximation)
- Support for causal masking without materializing the full attention matrix

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
(Dao et al., 2022) and "FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning" (Dao, 2023)

Args:
    embed_dim: Total dimension of the model
    num_heads: Number of parallel attention heads
    dropout: Dropout probability on attention weights (default: 0.0)
    causal: If True, applies causal masking for autoregressive models (default: False)
    use_flash_attn: Force enable/disable flash attention. If None, auto-detect (default: None)

Example:
    >>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
    >>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
    >>> output = attn(x)
    >>> output.shape
    torch.Size([2, 128, 512])

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes attention using Flash Attention when available.
- `extra_repr(self) -> str`
  String representation of the module.
- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Forward pass with Flash Attention and RoPE.

### `GroupedQueryAttention(embed_dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int, head_dim: int | None = None, dropout: float = 0.0, rope_theta: float = 10000.0, use_bias: bool = False, qkv_bias: bool = False, use_qk_norm: bool = False, rms_norm_eps: float = 1e-06, attention_scale: float | None = None, attn_logit_softcap: float | None = None)`

Grouped Query Attention (GQA) with Rotary Positional Embeddings.

GQA is a distinct attention mechanism where the number of Key/Value heads is smaller
than the number of Query heads. This reduces memory bandwidth usage during inference
(smaller KV cache) while maintaining performance close to Multi-Head Attention (MHA).

If num_kv_heads == num_heads, this is equivalent to MHA.
If num_kv_heads == 1, this is equivalent to Multi-Query Attention (MQA).

Args:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of Query heads.
    num_kv_heads (int): Number of Key/Value heads. Must divide num_heads.
    max_seq_len (int): Maximum sequence length for RoPE.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    rope_theta (float, optional): Base frequency for RoPE. Defaults to 10000.0.
    use_bias (bool, optional): Whether to use bias in linear projections. Defaults to False.

#### Methods

- `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Forward pass of Grouped Query Attention.

### `MultiHeadAttention(embed_dims: int, num_heads: int, dropout: float = 0.0, causal: bool = False)`

Implements Multi-Head Attention (MHA) as described in "Attention Is All You Need".

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Supports causal masking for autoregressive models.

Args:
    embed_dims (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
    causal (bool, optional): If True, applies a causal mask. Defaults to False.

Attributes:
    scale (float): Scaling factor (1 / sqrt(head_dim)).
    causal (bool): Whether to apply a causal mask.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the scaled dot-product attention.

### `MultiHeadAttentionwithALiBi(embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False, causal: bool = True, max_seq_len: int = 2048)`

Multi-Head Attention with ALiBi (Attention with Linear Biases).

ALiBi adds a static, non-learned bias to attention scores based on the distance between
query and key positions. This allows the model to extrapolate to longer sequence lengths
than seen during training.

Args:
    embed_dim (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    bias (bool, optional): Whether to use bias in linear projections. Defaults to False.
    causal (bool, optional): Whether to apply causal masking logic. Defaults to True.
    max_seq_len (int, optional): Max sequence length for precomputing ALiBi bias. Defaults to 2048.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes attention scores with ALiBi bias.

### `MultiHeadAttentionwithRoPE(embed_dims: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True, rope_theta: float = 10000.0)`

Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Uses RoPE for positional information.

Args:
    embed_dims (int): Total dimension of the model.
    num_heads (int): Number of parallel attention heads.
    max_seq_len (int): Maximum sequence length.
    dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
    causal (bool, optional): If True, applies a causal mask. Defaults to False.

Attributes:
    scale (float): Scaling factor (1 / sqrt(head_dim)).
    causal (bool): Whether to apply a causal mask.

#### Methods

- `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
  Computes the scaled dot-product attention suited for RoPE.
