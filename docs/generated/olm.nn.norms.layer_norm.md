# olm.nn.norms.layer_norm

### Classes

| [`LayerNorm`](#olm.nn.norms.layer_norm.LayerNorm)(\*args, \*\*kwargs)   | Layer Normalization layer.   |
|-------------------------------------------------------------------------|------------------------------|

### *class* olm.nn.norms.layer_norm.LayerNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.norms.layer_norm.NormBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for normalization layers (e.g., LayerNorm, RMSNorm).

Standardizes the interface for normalization, ensuring all implementations
handle model dimension, device, and dtype consistently.

#### d_model

The dimension of the input features to normalize.

* **Type:**
  int

#### device

The device the module is on.

* **Type:**
  torch.device, optional

#### dtype

The data type of the module parameters.

* **Type:**
  torch.dtype

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Apply normalization to the input tensor.
