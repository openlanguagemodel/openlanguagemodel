# `olm.nn.embeddings.positional.rope`

## Classes

### `PartialRotaryPositionalEmbedding(head_dim: int, rotary_percentage: float = 0.5, base: int = 10000, max_seq_len: int = 2048)`

Partial Rotary Positional Embedding (LLaMA-style RoPE).

Only applies rotary embeddings to a fraction of the head dimensions,
leaving the remaining dimensions unchanged. This is the approach used
in models like LLaMA, where typically 25-50% of dimensions are rotated.

For example, with head_dim=128 and rotary_percentage=0.5, only the first
64 dimensions are rotated, while the last 64 dimensions pass through unchanged.

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply partial rotary positional embedding to input tensor x.

### `PartialScaledRotaryPositionalEmbedding(head_dim: int, rotary_percentage: float = 0.5, max_seq_len: int = 2048, base: int = 10000, scaling_type: Literal['linear', 'ntk', 'dynamic_ntk', 'yarn', 'xpos'] = 'linear', scaling_factor: float = 1.0, original_max_seq_len: int | None = None, yarn_alpha: float = 1.0, yarn_beta: float = 32.0, xpos_scale_base: int | None = None)`

Partial Rotary Positional Embedding with scaling support.

Combines partial RoPE (only rotating a fraction of dimensions) with
various scaling strategies for extended context lengths.

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply partial scaled rotary positional embedding to input tensor x.

### `RotaryPositionalEmbedding(head_dim: int, max_seq_len: int, base: int = 10000)`

Rotary Positional Embedding (RoPE) as described in
“RoFormer: Enhanced Transformer with Rotary Position Embedding” (arXiv 2104.09864).

This module precomputes sin/cos rotation frequencies for a given head‐dim, and then applies to
query/key representations via interleaving real/imag parts (or equivalently pairs of dims).

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply rotary positional embedding to input tensor x.

### `ScaledRotaryPositionalEmbedding(head_dim: int, max_seq_len: int = 2048, base: int = 10000, scaling_type: Literal['linear', 'ntk', 'dynamic_ntk', 'yarn', 'xpos'] = 'linear', scaling_factor: float = 1.0, original_max_seq_len: int | None = None, yarn_alpha: float = 1.0, yarn_beta: float = 32.0, xpos_scale_base: int | None = None)`

Scaled Rotary Positional Embedding with multiple scaling strategies.

Supports the following scaling methods for extending context length:
- 'linear': Linear position interpolation (Position Interpolation, arXiv:2306.15595)
- 'ntk': NTK-aware scaling (dynamically adjusts base frequency)
- 'dynamic_ntk': Dynamic NTK (adjusts base based on current sequence length)
- 'yarn': YaRN (Yet another RoPE extensioN method, arXiv:2309.00071)
- 'xpos': XPos (exponential decay for better extrapolation, arXiv:2212.10554)

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply scaled rotary positional embedding to input tensor x.
