# `olm.nn.structure.combinators.residual`

Source: [`src/olm/nn/structure/combinators/residual.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/residual.py#L1)

## Classes

### `Residual(block: torch.nn.modules.module.Module)`

**Bases:** `olm.nn.structure.combinators.base.BaseCombinator`

Source: [`src/olm/nn/structure/combinators/residual.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/residual.py#L5)

Residual wrapper that adds the block output to its input.

**Parameters**

- `block`: Module applied to the input before residual addition.

**Attributes**

- `block`: Module used for the residual transformation.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/residual.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/residual.py#L26)

Apply the block and add the result to the input.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor with residual connection applied.
