# `olm.nn.structure.combinators.repeat`

Source: [`src/olm/nn/structure/combinators/repeat.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/repeat.py#L1)

## Classes

### `Repeat(module_func: Callable[[], torch.nn.modules.module.Module], num_repeat: int)`

**Bases:** `olm.nn.structure.combinators.base.BaseCombinator`

Source: [`src/olm/nn/structure/combinators/repeat.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/repeat.py#L6)

Repeat a module a fixed number of times in sequence.

The module function should return a new module instance each call. It is
used to build ``stack`` during initialization and is not needed for forward
passes after the modules have been created.

**Parameters**

- `module_func`: Callable returning a new module instance.
- `num_repeat`: Number of times to repeat the module.

**Attributes**

- `num_repeat`: Number of repeats.
- `stack`: ModuleList containing the repeated modules.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/structure/combinators/repeat.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/combinators/repeat.py#L42)

Apply the repeated modules in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all repeats.
