# `olm.nn.structure.combinators`

Source: [`src/olm/nn/structure/combinators/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/__init__.py#L1)

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

### `Repeat(module_func: Callable[[], torch.nn.modules.module.Module], num_repeat: int)`

**Bases:** `olm.nn.structure.combinators.base.BaseCombinator`

Source: [`src/olm/nn/structure/combinators/repeat.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/repeat.py#L6)

Repeat a module a fixed number of times in sequence.

The module function should return a new module instance each call. It is
used to build ``stack`` during initialization and is not needed for forward
passes after the modules have been created.

Args:
    module_func: Callable returning a new module instance.
    num_repeat: Number of times to repeat the module.

Attributes:
    num_repeat: Number of repeats.
    stack: ModuleList containing the repeated modules.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/repeat.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/repeat.py#L42)

Apply the repeated modules in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all repeats.

### `Residual(block: torch.nn.modules.module.Module)`

**Bases:** `olm.nn.structure.combinators.base.BaseCombinator`

Source: [`src/olm/nn/structure/combinators/residual.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/residual.py#L5)

Residual wrapper that adds the block output to its input.

Args:
    block: Module applied to the input before residual addition.

Attributes:
    block: Module used for the residual transformation.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/residual.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/residual.py#L26)

Apply the block and add the result to the input.

Args:
    x: Input tensor.

Returns:
    Output tensor with residual connection applied.
