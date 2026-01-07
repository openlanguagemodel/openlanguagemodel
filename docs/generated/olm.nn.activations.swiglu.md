# olm.nn.activations.swiglu

### Classes

| [`SwiGLU`](#olm.nn.activations.swiglu.SwiGLU)(\*args, \*\*kwargs)   | SwiGLU activation function.   |
|---------------------------------------------------------------------|-------------------------------|

### *class* olm.nn.activations.swiglu.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.swiglu.SwiGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

SwiGLU activation function.

Implements the SwiGLU activation as described in “GLU Variants Improve Transformer”.
It applies the SiLU activation to one half of the input (the gate) and multiplies it
by the other half (the value).

Equation:
: SwiGLU(x, W, V) = Swish_1(xW) \* (xV)
  Here, we assume the input x is already projected/concatenated such that we chunk it.
  So: SwiGLU(x) = (x_1 \* SiLU(x_2)) where x = [x_1, x_2]

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of SwiGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor. Expected to have an even last dimension size.
* **Returns:**
  Output tensor with half the last dimension of the input.
* **Return type:**
  torch.Tensor
