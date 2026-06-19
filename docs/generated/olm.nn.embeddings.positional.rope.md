# `olm.nn.embeddings.positional.rope`

Source: [`src/olm/nn/embeddings/positional/rope.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L1)

## Classes

### `PartialRotaryPositionalEmbedding(head_dim: int, rotary_percentage: float = 0.5, base: int = 10000, max_seq_len: int = 2048)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/rope.py:112`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L112)

Partial Rotary Positional Embedding (LLaMA-style RoPE).

Only applies rotary embeddings to a fraction of the head dimensions,
leaving the remaining dimensions unchanged. This is the approach used
in models like LLaMA, where typically 25-50% of dimensions are rotated.

For example, with head_dim=128 and rotary_percentage=0.5, only the first
64 dimensions are rotated, while the last 64 dimensions pass through unchanged.

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/rope.py:183`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L183)

Apply partial rotary positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, num_heads, head_dim)
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
        If None, assumes positions are 0..seq_len-1 for each batch.

Returns:
    Tensor of same shape as x, with partial RoPE applied.

### `PartialScaledRotaryPositionalEmbedding(head_dim: int, rotary_percentage: float = 0.5, max_seq_len: int = 2048, base: int = 10000, scaling_type: Literal['linear', 'ntk', 'dynamic_ntk', 'yarn', 'xpos'] = 'linear', scaling_factor: float = 1.0, original_max_seq_len: int | None = None, yarn_alpha: float = 1.0, yarn_beta: float = 32.0, xpos_scale_base: int | None = None)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/rope.py:478`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L478)

Partial Rotary Positional Embedding with scaling support.

Combines partial RoPE (only rotating a fraction of dimensions) with
various scaling strategies for extended context lengths.

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/rope.py:624`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L624)

Apply partial scaled rotary positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, num_heads, head_dim)
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.

Returns:
    Tensor of same shape as x, with partial scaled RoPE applied.

### `RotaryPositionalEmbedding(head_dim: int, max_seq_len: int, base: int = 10000)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/rope.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L8)

Rotary Positional Embedding (RoPE) as described in
“RoFormer: Enhanced Transformer with Rotary Position Embedding” (arXiv 2104.09864).

This module precomputes sin/cos rotation frequencies for a given head‐dim, and then applies to
query/key representations via interleaving real/imag parts (or equivalently pairs of dims).

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/rope.py:54`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L54)

Apply rotary positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, num_heads, head_dim)
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
        If None, assumes positions are 0..seq_len-1 for each batch.

Returns:
    Tensor of same shape as x, with RoPE applied.

### `ScaledRotaryPositionalEmbedding(head_dim: int, max_seq_len: int = 2048, base: int = 10000, scaling_type: Literal['linear', 'ntk', 'dynamic_ntk', 'yarn', 'xpos'] = 'linear', scaling_factor: float = 1.0, original_max_seq_len: int | None = None, yarn_alpha: float = 1.0, yarn_beta: float = 32.0, xpos_scale_base: int | None = None)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/rope.py:250`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L250)

Scaled Rotary Positional Embedding with multiple scaling strategies.

Supports the following scaling methods for extending context length:
- 'linear': Linear position interpolation (Position Interpolation, arXiv:2306.15595)
- 'ntk': NTK-aware scaling (dynamically adjusts base frequency)
- 'dynamic_ntk': Dynamic NTK (adjusts base based on current sequence length)
- 'yarn': YaRN (Yet another RoPE extensioN method, arXiv:2309.00071)
- 'xpos': XPos (exponential decay for better extrapolation, arXiv:2212.10554)

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/rope.py:398`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/rope.py#L398)

Apply scaled rotary positional embedding to input tensor x.

Args:
    x: shape (batch_size, seq_len, num_heads, head_dim)
    seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
        If None, assumes positions are 0..seq_len-1 for each batch.

Returns:
    Tensor of same shape as x, with scaled RoPE applied.
