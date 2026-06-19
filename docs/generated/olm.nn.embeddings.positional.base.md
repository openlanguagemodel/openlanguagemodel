# `olm.nn.embeddings.positional.base`

## Classes

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
