# `olm.nn.embeddings.positional`

Source: [`src/olm/nn/embeddings/positional/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/__init__.py#L1)

## Classes

### `ALiBiPositionalBias(num_heads: int, max_seq_len: int = 2048)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/alibi.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/alibi.py#L9)

Attention with Linear Biases (ALiBi) as described in
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
(arXiv 2108.12409).

Instead of adding positional information to embeddings, ALiBi adds a bias to attention scores
that is proportional to the distance between query and key positions. This allows the model
to extrapolate to longer sequences than seen during training.

The bias is computed as: ``bias[i,j] = -m * |i - j|``
where m is a head-specific slope.

#### Methods

##### `forward(self, seq_len_q: int, seq_len_k: int, device: torch.device | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/alibi.py:85`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/alibi.py#L85)

Get ALiBi bias for the given query and key sequence lengths.

**Parameters**

- `seq_len_q`: length of query sequence
- `seq_len_k`: length of key sequence (usually same as seq_len_q)
- `device`: device to place the bias tensor on

**Returns**

Bias tensor of shape (1, num_heads, seq_len_q, seq_len_k)
This should be added to attention scores before softmax.

### `AbsolutePositionalEmbedding(max_seq_len: int, embed_dim: int, dropout: float = 0.0)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/absolute.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/absolute.py#L8)

Absolute (Learned) Positional Embedding.

This is the standard positional embedding used in the original Transformer paper
and models like GPT-2. It learns a separate embedding vector for each position
in the sequence, up to a maximum sequence length.

These embeddings are typically added to token embeddings before passing through
the transformer blocks.

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/absolute.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/absolute.py#L34)

Apply absolute positional embedding to input tensor x.

**Parameters**

- `x`: shape (batch_size, seq_len, embed_dim) - token embeddings
- `seq_positions`: optional tensor of shape (batch_size, seq_len) with position indices. If None, assumes positions are 0..seq_len-1 for each batch.

**Returns**

Tensor of same shape as x, with positional embeddings added.

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

**Parameters**

- `x`: shape (batch_size, seq_len, num_heads, head_dim)
- `seq_positions`: optional tensor of shape (batch_size, seq_len) with position indices. If None, assumes positions are 0..seq_len-1 for each batch.

**Returns**

Tensor of same shape as x, with partial RoPE applied.

### `PositionalEmbeddingBase(*args: Any, **kwargs: Any) -> None`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/embeddings/positional/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/base.py#L8)

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

##### `extra_repr(self) -> str`

Source: [`src/olm/nn/embeddings/positional/base.py:40`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/base.py#L40)

String representation of the module for debugging.

Override this in subclasses to provide useful information.

##### `forward(self, *args, **kwargs) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/base.py:25`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/base.py#L25)

Apply positional information to input tensor(s).

The signature and behavior of this method varies by implementation:
- Some add to embeddings (Absolute, Sinusoidal)
- Some rotate representations (RoPE)
- Some return bias to add to attention scores (ALiBi)

**Returns**

Transformed tensor(s) with positional information applied

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

**Parameters**

- `x`: shape (batch_size, seq_len, num_heads, head_dim)
- `seq_positions`: optional tensor of shape (batch_size, seq_len) with position indices. If None, assumes positions are 0..seq_len-1 for each batch.

**Returns**

Tensor of same shape as x, with RoPE applied.

### `SinusoidalPositionalEmbedding(embed_dim: int, max_seq_len: int = 5000, base: int = 10000, dropout: float = 0.0)`

**Bases:** `olm.nn.embeddings.positional.base.PositionalEmbeddingBase`

Source: [`src/olm/nn/embeddings/positional/sinusoidal.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/sinusoidal.py#L9)

Sinusoidal Positional Embedding as described in
"Attention Is All You Need" (Vaswani et al., 2017).

Uses fixed sine and cosine functions of different frequencies to encode positions.
Unlike learned embeddings, these are deterministic and can extrapolate to longer
sequences than seen during training.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#### Methods

##### `forward(self, x: torch.Tensor, seq_positions: torch.LongTensor | None = None) -> torch.Tensor`

Source: [`src/olm/nn/embeddings/positional/sinusoidal.py:80`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/sinusoidal.py#L80)

Apply sinusoidal positional embedding to input tensor x.

**Parameters**

- `x`: shape (batch_size, seq_len, embed_dim) - token embeddings
- `seq_positions`: optional tensor of shape (batch_size, seq_len) with position indices. If None, assumes positions are 0..seq_len-1 for each batch.

**Returns**

Tensor of same shape as x, with positional embeddings added.
