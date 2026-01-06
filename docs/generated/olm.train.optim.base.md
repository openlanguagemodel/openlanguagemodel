# olm.train.optim.base

### Classes

| [`OptimizerBase`](#olm.train.optim.base.OptimizerBase)(\*args, \*\*kwargs)   | Abstract base class for all optimizers in the OLM framework.   |
|------------------------------------------------------------------------------|----------------------------------------------------------------|

### *class* olm.train.optim.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.train.optim.base.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.train.optim.base.OptimizerBase(\*args: [Any](#olm.train.optim.base.Any), \*\*kwargs: [Any](#olm.train.optim.base.Any))

Bases: `Optimizer`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch’s Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### extra_repr() → str

String representation of the optimizer for debugging.

Override this in subclasses to provide useful information.

#### load_state_dict(state_dict: Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)])

Loads the optimizer state.

* **Parameters:**
  **state_dict** – optimizer state. Should be an object returned
  from a call to state_dict().

#### state_dict() → Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)]

Returns the state of the optimizer as a dict.

It contains two entries:

- `state`: dict holding current optimization state. Its content
  differs between optimizer classes.
- `param_groups`: list containing all parameter groups where each
  parameter group is a dict.

* **Returns:**
  Dictionary containing optimizer state

#### *abstractmethod* step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
  Some optimization algorithms (e.g., L-BFGS) require multiple
  evaluations of the loss function.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized tensors to zero or None.

* **Parameters:**
  **set_to_none** – Instead of setting to zero, set the grads to None.
  This is more memory efficient and can slightly improve performance.
  Default: True

### olm.train.optim.base.abstractmethod(funcobj)

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
