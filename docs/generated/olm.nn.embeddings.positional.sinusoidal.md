# `olm.nn.embeddings.positional.sinusoidal`

## Classes

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
