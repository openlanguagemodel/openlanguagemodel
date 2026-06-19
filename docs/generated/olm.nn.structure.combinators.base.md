# `olm.nn.structure.combinators.base`

## Classes

### `BaseCombinator()`

Abstract base class for combinator modules.

Subclasses implement ``forward`` to define how inputs are combined.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Compute the combinator output from an input tensor.
