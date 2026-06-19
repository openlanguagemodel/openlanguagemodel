# `olm.nn.structure.combinators.parallel`

Source: [`src/olm/nn/structure/combinators/parallel.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/parallel.py#L1)

## Classes

### `Parallel(blocks: List[torch.nn.modules.module.Module], merge: Callable = None, dim: int = -1)`

**Bases:** `olm.nn.structure.combinators.base.BaseCombinator`

Source: [`src/olm/nn/structure/combinators/parallel.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/parallel.py#L6)

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

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/parallel.py:37`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/parallel.py#L37)

Apply all blocks in parallel and merge their outputs.

Args:
    x: Input tensor.

Returns:
    Merged output tensor.
