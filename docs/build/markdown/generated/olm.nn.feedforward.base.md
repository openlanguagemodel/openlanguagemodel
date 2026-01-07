# olm.nn.feedforward.base

### Classes

| [`FeedForwardBase`](#olm.nn.feedforward.base.FeedForwardBase)(\*args, \*\*kwargs)   | Abstract base class for feedforward networks in a transformer block.   |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### *class* olm.nn.feedforward.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.nn.feedforward.base.FeedForwardBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for feedforward networks in a transformer block.

Defines the interface for FFNs/MLPs. Subclasses must implement the forward method.

#### embed_dim

The input and output dimension.

* **Type:**
  int

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Forward pass of the feedforward network.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, embed_dim).
* **Returns:**
  Output tensor of shape (batch, seq_len, embed_dim).
* **Return type:**
  torch.Tensor

### olm.nn.feedforward.base.abstractmethod(funcobj)

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
