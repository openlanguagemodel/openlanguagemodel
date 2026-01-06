# olm.nn.activations.liglu

### Classes

| [`LiGLU`](#olm.nn.activations.liglu.LiGLU)(\*args, \*\*kwargs)   | LiGLU activation function.   |
|------------------------------------------------------------------|------------------------------|

### *class* olm.nn.activations.liglu.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.liglu.LiGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

LiGLU activation function.

Implements the LiGLU variant (Linear GLU).
LiGLU(x, W, V) = (xW) \* (xV)
Here: LiGLU(x) = gate \* value (No activation on gate)

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of LiGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor.
* **Returns:**
  Output tensor with half the last dimension.
* **Return type:**
  torch.Tensor
