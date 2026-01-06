# olm.nn.activations

### *class* olm.nn.activations.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.ELU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

ELU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.GELU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GELU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.GLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GLU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.GeGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.Identity(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Identity activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.LeakyReLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

LeakyReLU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.LiGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.Mish(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Mish activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.PReLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

PReLU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.ReGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

ReGLU activation function.

Implements the ReGLU variant from “GLU Variants Improve Transformer”.
ReGLU(x, W, V) = ReLU(xW) \* (xV)
Here: ReGLU(x) = ReLU(gate) \* value

* **Parameters:**
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of ReGLU.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor.
* **Returns:**
  Output tensor with half the last dimension.
* **Return type:**
  torch.Tensor

### *class* olm.nn.activations.ReLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

ReLU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.SELU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

SELU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.SiLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

SiLU (Swish) activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.Sigmoid(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Sigmoid activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.Softmax(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Softmax activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.Softplus(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Softplus activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### *class* olm.nn.activations.SwiGLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.Tanh(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Tanh activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.

### Modules

| [`base`](olm.nn.activations.base.md#module-olm.nn.activations.base)                   |    |
|---------------------------------------------------------------------------------------|----|
| [`clu`](olm.nn.activations.clu.md#module-olm.nn.activations.clu)                      |    |
| [`elu`](olm.nn.activations.elu.md#module-olm.nn.activations.elu)                      |    |
| [`geglu`](olm.nn.activations.geglu.md#module-olm.nn.activations.geglu)                |    |
| [`gelu`](olm.nn.activations.gelu.md#module-olm.nn.activations.gelu)                   |    |
| [`glu`](olm.nn.activations.glu.md#module-olm.nn.activations.glu)                      |    |
| [`identity`](olm.nn.activations.identity.md#module-olm.nn.activations.identity)       |    |
| [`leaky_relu`](olm.nn.activations.leaky_relu.md#module-olm.nn.activations.leaky_relu) |    |
| [`liglu`](olm.nn.activations.liglu.md#module-olm.nn.activations.liglu)                |    |
| [`mish`](olm.nn.activations.mish.md#module-olm.nn.activations.mish)                   |    |
| [`prelu`](olm.nn.activations.prelu.md#module-olm.nn.activations.prelu)                |    |
| [`reglu`](olm.nn.activations.reglu.md#module-olm.nn.activations.reglu)                |    |
| [`relu`](olm.nn.activations.relu.md#module-olm.nn.activations.relu)                   |    |
| [`selu`](olm.nn.activations.selu.md#module-olm.nn.activations.selu)                   |    |
| [`sigmoid`](olm.nn.activations.sigmoid.md#module-olm.nn.activations.sigmoid)          |    |
| [`silu`](olm.nn.activations.silu.md#module-olm.nn.activations.silu)                   |    |
| [`softmax`](olm.nn.activations.softmax.md#module-olm.nn.activations.softmax)          |    |
| [`softplus`](olm.nn.activations.softplus.md#module-olm.nn.activations.softplus)       |    |
| [`swiglu`](olm.nn.activations.swiglu.md#module-olm.nn.activations.swiglu)             |    |
| [`tanh`](olm.nn.activations.tanh.md#module-olm.nn.activations.tanh)                   |    |
