# olm.data.datasets.base_dataset

### Classes

| [`BaseTextDataset`](#olm.data.datasets.base_dataset.BaseTextDataset)(\*args, \*\*kwargs)   | Abstract base class for text-based streaming datasets.   |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------|

### *class* olm.data.datasets.base_dataset.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.data.datasets.base_dataset.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.data.datasets.base_dataset.BaseTextDataset(\*args: [Any](#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](#olm.data.datasets.base_dataset.Any))

Bases: `IterableDataset`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for text-based streaming datasets.

Handles tokenization buffering and sequence generation generically.
Subclasses must implement \_get_text_iterator to yield text chunks.

### *class* olm.data.datasets.base_dataset.Union

Bases: `object`

Represent a union type

E.g. for int | str

### olm.data.datasets.base_dataset.abstractmethod(funcobj)

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

### *class* olm.data.datasets.base_dataset.islice

Bases: `object`

islice(iterable, stop) –> islice object
islice(iterable, start, stop[, step]) –> islice object

Return an iterator whose next() method returns selected values from an
iterable.  If start is specified, will skip all preceding elements;
otherwise, start defaults to zero.  Step defaults to one.  If
specified as another value, step determines how many values are
skipped between successive calls.  Works like a slice() on a list
but returns an iterator.
