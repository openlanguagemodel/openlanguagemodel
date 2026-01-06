# olm.nn.activations.tanh

### Classes

| [`Tanh`](#olm.nn.activations.tanh.Tanh)(\*args, \*\*kwargs)   | Tanh activation wrapper.   |
|---------------------------------------------------------------|----------------------------|

### *class* olm.nn.activations.tanh.ActivationBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.nn.activations.tanh.Tanh(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`ActivationBase`](olm.nn.activations.base.md#olm.nn.activations.base.ActivationBase)

Tanh activation wrapper.

#### forward(x: torch.Tensor) → torch.Tensor

Apply activation to `x`.
