# `olm.nn.structure.combinators.residual`

## Classes

### `Residual(block: torch.nn.modules.module.Module)`

Residual wrapper that adds the block output to its input.

Args:
    block: Module applied to the input before residual addition.

Attributes:
    block: Module used for the residual transformation.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply the block and add the result to the input.
