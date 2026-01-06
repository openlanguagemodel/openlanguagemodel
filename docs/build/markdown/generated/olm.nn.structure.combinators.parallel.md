# olm.nn.structure.combinators.parallel

### Classes

| [`Parallel`](#olm.nn.structure.combinators.parallel.Parallel)(\*args, \*\*kwargs)   | Apply multiple blocks to the same input and merge their outputs.   |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------|

### *class* olm.nn.structure.combinators.parallel.BaseCombinator(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for combinator modules.

Subclasses implement `forward` to define how inputs are combined.

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Compute the combinator output from an input tensor.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor produced by the combinator.

### *class* olm.nn.structure.combinators.parallel.Parallel(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.structure.combinators.parallel.Union

Bases: `object`

Represent a union type

E.g. for int | str
