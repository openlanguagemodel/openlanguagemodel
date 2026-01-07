# olm.nn.structure.combinators

### *class* olm.nn.structure.combinators.BaseCombinator(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for combinator modules.

Subclasses implement `forward` to define how inputs are combined.

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Compute the combinator output from an input tensor.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor produced by the combinator.

### *class* olm.nn.structure.combinators.Parallel(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Apply multiple blocks to the same input and merge their outputs.

The merge function takes a list of tensors and a dimension argument.

* **Parameters:**
  * **blocks** – Modules applied in parallel to the same input.
  * **merge** – Function that combines the list of outputs and a dimension.
  * **dim** – Dimension used by the merge function when applicable.

#### blocks

ModuleList storing the parallel blocks.

#### merge

Merge function used to combine outputs.

#### dim

Dimension passed to the merge function.

#### forward(x: torch.Tensor) → torch.Tensor

Apply all blocks in parallel and merge their outputs.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Merged output tensor.

### *class* olm.nn.structure.combinators.Repeat(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.structure.combinators.Residual(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Residual wrapper that adds the block output to its input.

* **Parameters:**
  **block** – Module applied to the input before residual addition.

#### block

Module used for the residual transformation.

#### forward(x: torch.Tensor) → torch.Tensor

Apply the block and add the result to the input.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor with residual connection applied.

### Modules

| [`base`](olm.nn.structure.combinators.base.md#module-olm.nn.structure.combinators.base)             |    |
|-----------------------------------------------------------------------------------------------------|----|
| [`parallel`](olm.nn.structure.combinators.parallel.md#module-olm.nn.structure.combinators.parallel) |    |
| [`repeat`](olm.nn.structure.combinators.repeat.md#module-olm.nn.structure.combinators.repeat)       |    |
| [`residual`](olm.nn.structure.combinators.residual.md#module-olm.nn.structure.combinators.residual) |    |
