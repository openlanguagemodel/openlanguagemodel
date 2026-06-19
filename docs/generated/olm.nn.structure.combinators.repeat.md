# `olm.nn.structure.combinators.repeat`

## Classes

### `Repeat(module_func: Callable[[], torch.nn.modules.module.Module], num_repeat: int)`

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

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Apply the repeated modules in sequence.
