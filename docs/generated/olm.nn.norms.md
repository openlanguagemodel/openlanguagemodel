# olm.nn.norms

### *class* olm.nn.norms.LayerNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.norms.RMSNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`NormBase`](olm.nn.norms.base.md#olm.nn.norms.base.NormBase)

RMSNorm (Root Mean Square Layer Normalization) layer.

Implements RMSNorm as described in “Root Mean Square Layer Normalization” ([https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)).
A simplified version of LayerNorm that scales invariance properties.

* **Parameters:**
  * **d_model** (*int*) – The dimension of the model to normalize.
  * **eps** (*float* *,* *optional*) – Small constant for numerical stability. Defaults to 1e-5.
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### weight

Learnable scale parameter.

* **Type:**
  nn.Parameter

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of RMSNorm.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch_size, sequence_length, d_model).
* **Returns:**
  Normalized output tensor of the same shape.
* **Return type:**
  torch.Tensor

### Modules

| [`base`](olm.nn.norms.base.md#module-olm.nn.norms.base)                   |    |
|---------------------------------------------------------------------------|----|
| [`layer_norm`](olm.nn.norms.layer_norm.md#module-olm.nn.norms.layer_norm) |    |
| [`rms_norm`](olm.nn.norms.rms_norm.md#module-olm.nn.norms.rms_norm)       |    |
