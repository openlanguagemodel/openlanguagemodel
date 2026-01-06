# olm.nn.activations.geglu

### Classes

| [`GeGLU`](#olm.nn.activations.geglu.GeGLU)(\*args, \*\*kwargs)   | GeGLU activation function.   |
|------------------------------------------------------------------|------------------------------|

### *class* olm.nn.activations.geglu.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all activation functions.

Ensures a consistent interface for activation layers, handling device and dtype
initialization. Subclasses must implement the forward method.

#### device

The device the module is on.

* **Type:**
  torch.device, optional

#### dtype

The data type of the module parameters.

* **Type:**
  torch.dtype

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.geglu.GeGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GeGLU activation function.

Implements the GeGLU variant from “GLU Variants Improve Transformer”.
GeGLU(x, W, V) = GELU(xW) \* (xV)
Here: GeGLU(x) = GELU(gate) \* value

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of GeGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor.
* **Returns:**
  Output tensor with half the last dimension.
* **Return type:**
  torch.Tensor
