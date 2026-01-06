# olm.nn.structure.combinators.residual

### Classes

| [`Residual`](#olm.nn.structure.combinators.residual.Residual)(\*args, \*\*kwargs)   | Residual wrapper that adds the block output to its input.   |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------|

### *class* olm.nn.structure.combinators.residual.BaseCombinator(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for combinator modules.

Subclasses implement `forward` to define how inputs are combined.

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Compute the combinator output from an input tensor.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor produced by the combinator.

### *class* olm.nn.structure.combinators.residual.Residual(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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
