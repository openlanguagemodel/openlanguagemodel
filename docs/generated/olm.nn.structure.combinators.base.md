# `olm.nn.structure.combinators.base`

Source: [`src/olm/nn/structure/combinators/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/base.py#L1)

## Classes

### `BaseCombinator()`

**Bases:** `Module`, `ABC`

Source: [`src/olm/nn/structure/combinators/base.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/base.py#L5)

Abstract base class for combinator modules.

Subclasses implement ``forward`` to define how inputs are combined.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/base.py:15`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/base.py#L15)

Compute the combinator output from an input tensor.

Args:
    x: Input tensor.

Returns:
    Output tensor produced by the combinator.
