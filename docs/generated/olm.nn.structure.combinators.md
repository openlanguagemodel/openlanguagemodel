# `olm.nn.structure.combinators`

## Classes

### `BaseCombinator()`

Abstract base class for combinator modules.

Subclasses implement ``forward`` to define how inputs are combined.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Compute the combinator output from an input tensor.

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

### `Repeat(module_func: Callable[[], torch.nn.modules.module.Module], num_repeat: int)`

Repeat a module a fixed number of times in sequence.

The module function should return a new module instance each call.

Args:
    module_func: Callable returning a new module instance.
    num_repeat: Number of times to repeat the module.

Attributes:
    module: Factory callable used to create new modules.
    num_repeat: Number of repeats.
    stack: ModuleList containing the repeated modules.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply the repeated modules in sequence.

### `Residual(block: torch.nn.modules.module.Module)`

Residual wrapper that adds the block output to its input.

Args:
    block: Module applied to the input before residual addition.

Attributes:
    block: Module used for the residual transformation.

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply the block and add the result to the input.
