# `olm.nn.embeddings.positional`

## Classes

### `ALiBiPositionalBias(num_heads: int, max_seq_len: int = 2048)`

Attention with Linear Biases (ALiBi) as described in
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
(arXiv 2108.12409).

Instead of adding positional information to embeddings, ALiBi adds a bias to attention scores
that is proportional to the distance between query and key positions. This allows the model
to extrapolate to longer sequences than seen during training.

The bias is computed as: ``bias[i,j] = -m * |i - j|``
where m is a head-specific slope.

#### Methods

- `forward(self, seq_len_q: int, seq_len_k: int, device: torch.device | None = None) -> torch.Tensor`
  Get ALiBi bias for the given query and key sequence lengths.

### `AbsolutePositionalEmbedding(max_seq_len: int, embed_dim: int, dropout: float = 0.0)`

Absolute (Learned) Positional Embedding.

This is the standard positional embedding used in the original Transformer paper
and models like GPT-2. It learns a separate embedding vector for each position
in the sequence, up to a maximum sequence length.

These embeddings are typically added to token embeddings before passing through
the transformer blocks.

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply absolute positional embedding to input tensor x.

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

### `PositionalEmbeddingBase(*args: Any, **kwargs: Any) -> None`

Abstract base class for all positional embedding implementations.

Positional embeddings add information about token positions in a sequence
to help the model understand order and relative positions. Different positional
embedding strategies have different properties:

- Learned (Absolute): Simple, effective, but limited to max_seq_len
- Sinusoidal: Deterministic, can extrapolate to longer sequences
- RoPE: Applied to Q/K directly, enables relative position modeling
- ALiBi: Adds bias to attention scores, excellent extrapolation

All positional embedding implementations should inherit from this base class
and implement the forward method.

#### Methods

- `extra_repr(self) -> str`
  String representation of the module for debugging.
- `forward(self, *args, **kwargs) -> torch.Tensor`
  Apply positional information to input tensor(s).

### `RotaryPositionalEmbedding(head_dim: int, max_seq_len: int, base: int = 10000)`

Rotary Positional Embedding (RoPE) as described in
“RoFormer: Enhanced Transformer with Rotary Position Embedding” (arXiv 2104.09864).

This module precomputes sin/cos rotation frequencies for a given head‐dim, and then applies to
query/key representations via interleaving real/imag parts (or equivalently pairs of dims).

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply rotary positional embedding to input tensor x.

### `SinusoidalPositionalEmbedding(embed_dim: int, max_seq_len: int = 5000, base: int = 10000, dropout: float = 0.0)`

Sinusoidal Positional Embedding as described in
"Attention Is All You Need" (Vaswani et al., 2017).

Uses fixed sine and cosine functions of different frequencies to encode positions.
Unlike learned embeddings, these are deterministic and can extrapolate to longer
sequences than seen during training.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#### Methods

- `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`
  Apply sinusoidal positional embedding to input tensor x.
