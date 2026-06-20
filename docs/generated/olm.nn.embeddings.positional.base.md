# `olm.nn.embeddings.positional.base`

Source: [`src/olm/nn/embeddings/positional/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/embeddings/positional/base.py#L1)

## Classes

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
