# `olm.nn.structure.combinators.repeat`

## Classes

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
