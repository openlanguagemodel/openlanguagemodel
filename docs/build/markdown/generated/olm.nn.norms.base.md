# olm.nn.norms.base

### Classes

| [`NormBase`](#olm.nn.norms.base.NormBase)(\*args, \*\*kwargs)   | Abstract base class for normalization layers (e.g., LayerNorm, RMSNorm).   |
|-----------------------------------------------------------------|----------------------------------------------------------------------------|

### *class* olm.nn.norms.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.nn.norms.base.NormBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### olm.nn.norms.base.abstractmethod(funcobj)

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
‘super’ call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

> class C(metaclass=ABCMeta):
> : @abstractmethod
>   def my_abstract_method(self, arg1, arg2, argN):
>   <br/>
>   > …
