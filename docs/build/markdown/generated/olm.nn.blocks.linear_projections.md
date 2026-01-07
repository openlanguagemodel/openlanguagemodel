# olm.nn.blocks.linear_projections

### Classes

| [`QKVProjection`](#olm.nn.blocks.linear_projections.QKVProjection)(\*args, \*\*kwargs)   | Computes Query, Key, and Value projections for attention mechanisms.   |
|------------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### *class* olm.nn.blocks.linear_projections.Linear(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Linear`

#### forward(x)

### *class* olm.nn.blocks.linear_projections.QKVProjection(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Computes Query, Key, and Value projections for attention mechanisms.

Applies three separate linear transformations to the input to generate Q, K, and V tensors.
Supports various weight initialization schemes.

#### W_q

Linear layer for Query projection.

* **Type:**
  [Linear](#olm.nn.blocks.linear_projections.Linear)

#### W_k

Linear layer for Key projection.

* **Type:**
  [Linear](#olm.nn.blocks.linear_projections.Linear)

#### W_v

Linear layer for Value projection.

* **Type:**
  [Linear](#olm.nn.blocks.linear_projections.Linear)

#### forward(x: torch.Tensor) → Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

Performs the Q, K, V projections.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, dim_in).
* **Returns:**
  A tuple containing (Q, K, V) tensors.
* **Return type:**
  tuple[torch.Tensor, torch.Tensor, torch.Tensor]
