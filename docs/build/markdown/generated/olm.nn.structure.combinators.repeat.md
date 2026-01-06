# olm.nn.structure.combinators.repeat

### Classes

| [`Repeat`](#olm.nn.structure.combinators.repeat.Repeat)(\*args, \*\*kwargs)   | Repeat a module a fixed number of times in sequence.   |
|-------------------------------------------------------------------------------|--------------------------------------------------------|

### *class* olm.nn.structure.combinators.repeat.BaseCombinator(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for combinator modules.

Subclasses implement `forward` to define how inputs are combined.

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Compute the combinator output from an input tensor.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor produced by the combinator.

### *class* olm.nn.structure.combinators.repeat.Repeat(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Repeat a module a fixed number of times in sequence.

The module function should return a new module instance each call.

* **Parameters:**
  * **module_func** – Callable returning a new module instance.
  * **num_repeat** – Number of times to repeat the module.

#### module

Factory callable used to create new modules.

#### num_repeat

Number of repeats.

#### stack

ModuleList containing the repeated modules.

#### forward(x: torch.Tensor) → torch.Tensor

Apply the repeated modules in sequence.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor after all repeats.
