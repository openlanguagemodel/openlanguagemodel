# olm.nn.activations.glu

### Classes

| [`GLU`](#olm.nn.activations.glu.GLU)(\*args, \*\*kwargs)   | GLU activation wrapper.   |
|------------------------------------------------------------|---------------------------|

### *class* olm.nn.activations.glu.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.glu.GLU(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

GLU activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.
