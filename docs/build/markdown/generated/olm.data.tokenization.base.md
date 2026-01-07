# olm.data.tokenization.base

### Classes

| [`TokenizerBase`](#olm.data.tokenization.base.TokenizerBase)()   | Abstract base class for all tokenizers in OLM.   |
|------------------------------------------------------------------|--------------------------------------------------|

### *class* olm.data.tokenization.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.data.tokenization.base.TokenizerBase

Bases: [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement encode and decode methods.

#### *abstractmethod* decode(tokens: torch.Tensor) → str

Converts a sequence of token IDs back into a text string.

* **Parameters:**
  **tokens** (*torch.Tensor*) – A 1D tensor or list of token IDs.
* **Returns:**
  The decoded text string.
* **Return type:**
  str

#### *abstractmethod* encode(text: str) → torch.Tensor

Converts a text string into a sequence of token IDs.

* **Parameters:**
  **text** (*str*) – The input text to tokenize.
* **Returns:**
  A 1D tensor containing the token IDs.
* **Return type:**
  torch.Tensor

### olm.data.tokenization.base.abstractmethod(funcobj)

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
