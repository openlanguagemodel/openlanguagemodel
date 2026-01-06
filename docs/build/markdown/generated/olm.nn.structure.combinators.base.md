# olm.nn.structure.combinators.base

### Classes

| [`BaseCombinator`](#olm.nn.structure.combinators.base.BaseCombinator)(\*args, \*\*kwargs)   | Abstract base class for combinator modules.   |
|---------------------------------------------------------------------------------------------|-----------------------------------------------|

### *class* olm.nn.structure.combinators.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.nn.structure.combinators.base.BaseCombinator(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for combinator modules.

Subclasses implement `forward` to define how inputs are combined.

#### *abstractmethod* forward(x: torch.Tensor) → torch.Tensor

Compute the combinator output from an input tensor.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor produced by the combinator.

### olm.nn.structure.combinators.base.abstractmethod(funcobj)

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
