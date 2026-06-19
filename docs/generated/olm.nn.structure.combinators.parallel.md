# `olm.nn.structure.combinators.parallel`

## Classes

### `Parallel(blocks: List[torch.nn.modules.module.Module], merge: Callable = None, dim: int = -1)`

Apply multiple blocks to the same input and merge their outputs.

The merge function takes a list of tensors and a dimension argument.

Args:
    blocks: Modules applied in parallel to the same input.
    merge: Function that combines the list of outputs and a dimension.
    dim: Dimension used by the merge function when applicable.

Attributes:
    blocks: ModuleList storing the parallel blocks.
    merge: Merge function used to combine outputs.
    dim: Dimension passed to the merge function.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply all blocks in parallel and merge their outputs.
