# olm.nn.blocks.output_head

### Classes

| [`OutputHead`](#olm.nn.blocks.output_head.OutputHead)(\*args, \*\*kwargs)   | Final output projection layer for the Language Model.   |
|-----------------------------------------------------------------------------|---------------------------------------------------------|

### *class* olm.nn.blocks.output_head.Block(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.blocks.output_head.LayerNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`NormBase`](olm.nn.norms.base.md#olm.nn.norms.base.NormBase)

Layer Normalization layer.

Implements Layer Normalization as described in “Layer Normalization” ([https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)).
Normalizes the input across the features dimension.

* **Parameters:**
  * **d_model** (*int*) – The dimension of the model to normalize.
  * **eps** (*float* *,* *optional*) – Small constant for numerical stability. Defaults to 1e-5.
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### gamma

Learnable scale parameter.

* **Type:**
  nn.Parameter

#### beta

Learnable shift parameter.

* **Type:**
  nn.Parameter

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of LayerNorm.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch_size, sequence_length, d_model).
* **Returns:**
  Normalized output tensor of the same shape.
* **Return type:**
  torch.Tensor

### *class* olm.nn.blocks.output_head.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)

### *class* olm.nn.blocks.output_head.OutputHead(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`Block`](olm.nn.structure.block.md#olm.nn.structure.block.Block)

Final output projection layer for the Language Model.

Consists of a LayerNorm followed by a Linear projection to the vocabulary size.
Typical structure: LayerNorm -> Linear(vocab_size).

* **Parameters:**
  * **embed_dim** (*int*) – The dimension of the embedding space.
  * **vocab_size** (*int*) – The size of the vocabulary.
  * **bias** (*bool* *,* *optional*) – Whether to include bias in the linear layer. Defaults to False.

#### layers

The normalization and linear layers.

* **Type:**
  nn.ModuleList
