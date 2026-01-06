# olm.nn.structure

### *class* olm.nn.structure.Block(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Lightweight sequential container for composable submodules.

Similar to `nn.Sequential`, but exposes the underlying list for
inspection or dynamic manipulation by higher-level builders.

* **Parameters:**
  **blocks** – Ordered list of modules applied to the input in sequence.

#### blocks

ModuleList storing the ordered blocks.

#### forward(x: torch.Tensor) → torch.Tensor

Apply each block to the input in sequence.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor after all blocks have been applied.

### Modules

| [`block`](olm.nn.structure.block.md#module-olm.nn.structure.block)                   |    |
|--------------------------------------------------------------------------------------|----|
| [`combinators`](olm.nn.structure.combinators.md#module-olm.nn.structure.combinators) |    |
