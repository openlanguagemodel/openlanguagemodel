# `olm.nn.attention.flash`

Source: [`src/olm/nn/attention/flash.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L1)

## Classes

### `FlashAttention(embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, use_flash_attn: bool | None = None)`

**Bases:** `olm.nn.attention.base.AttentionBase`

Source: [`src/olm/nn/attention/flash.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L12)

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

**Parameters**

- `embed_dim`: Total dimension of the model
- `num_heads`: Number of parallel attention heads
- `dropout`: Dropout probability on attention weights (default: 0.0)
- `causal`: If True, applies causal masking for autoregressive models (default: False)
- `use_flash_attn`: Force enable/disable flash attention. If None, auto-detect (default: None)

**Example**

```python
attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
output = attn(x)
output.shape
torch.Size([2, 128, 512])
```

#### Methods

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/flash.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L73)

Computes attention using Flash Attention when available.

**Parameters**

- `q`: Query tensor [batch, heads, seq, head_dim]
- `k`: Key tensor [batch, heads, seq, head_dim]
- `v`: Value tensor [batch, heads, seq, head_dim]
- `mask`: Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]

**Returns**

Attention output [batch, heads, seq, head_dim]

##### `extra_repr(self) -> str`

Source: [`src/olm/nn/attention/flash.py:206`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L206)

String representation of the module.

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/flash.py:177`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L177)

Forward pass with Flash Attention.

**Parameters**

- `x`: Input tensor [batch, seq_len, embed_dim]
- `mask`: Optional attention mask

**Returns**

Output tensor [batch, seq_len, embed_dim]

### `FlashAttentionwithRoPE(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True, rope_theta: float = 10000.0, use_flash_attn: bool | None = None)`

**Bases:** `olm.nn.attention.base.AttentionwithRoPEBase`

Source: [`src/olm/nn/attention/flash.py:215`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L215)

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

**Parameters**

- `embed_dim`: Total dimension of the model
- `num_heads`: Number of parallel attention heads
- `dropout`: Dropout probability on attention weights (default: 0.0)
- `causal`: If True, applies causal masking for autoregressive models (default: False)
- `use_flash_attn`: Force enable/disable flash attention. If None, auto-detect (default: None)

**Example**

```python
attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
output = attn(x)
output.shape
torch.Size([2, 128, 512])
```

#### Methods

##### `compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/flash.py:286`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L286)

Computes attention using Flash Attention when available.

**Parameters**

- `q`: Query tensor [batch, heads, seq, head_dim]
- `k`: Key tensor [batch, heads, seq, head_dim]
- `v`: Value tensor [batch, heads, seq, head_dim]
- `mask`: Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]

**Returns**

Attention output [batch, heads, seq, head_dim]

##### `extra_repr(self) -> str`

Source: [`src/olm/nn/attention/flash.py:428`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L428)

String representation of the module.

##### `forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/attention/flash.py:390`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/attention/flash.py#L390)

Forward pass with Flash Attention and RoPE.

**Parameters**

- `x`: Input tensor [batch, seq_len, embed_dim]
- `mask`: Optional attention mask

**Returns**

Output tensor [batch, seq_len, embed_dim]
