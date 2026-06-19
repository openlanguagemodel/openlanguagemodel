# `olm.nn.embeddings.positional.alibi`

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
