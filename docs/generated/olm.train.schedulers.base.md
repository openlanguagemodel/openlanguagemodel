# olm.train.schedulers.base

Base learning rate scheduler for OLM.

### Classes

| [`SchedulerBase`](#olm.train.schedulers.base.SchedulerBase)(\*args, \*\*kwargs)   | Base class for all OLM learning rate schedulers.   |
|-----------------------------------------------------------------------------------|----------------------------------------------------|

### *class* olm.train.schedulers.base.ABC

Bases: `object`

Helper class that provides a standard way to create an ABC using
inheritance.

### *class* olm.train.schedulers.base.SchedulerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `_LRScheduler`, [`ABC`](#olm.train.schedulers.base.ABC)

Base class for all OLM learning rate schedulers.

This class extends PyTorch’s \_LRScheduler and provides a consistent
interface for implementing custom learning rate schedules. All OLM
schedulers should inherit from this class to maintain uniformity.

Subclasses must implement:
: - get_lr(): Compute the learning rate for the current step
  - \_get_closed_form_lr() (optional): Closed-form solution for efficiency

* **Parameters:**
  * **optimizer** – Wrapped PyTorch optimizer.
  * **last_epoch** – The index of the last epoch (default: -1).
  * **verbose** – If True, prints a message to stdout for each update (default: False).

### Example

```pycon
>>> class MyScheduler(SchedulerBase):
...     def __init__(self, optimizer, param, last_epoch=-1):
...         self.param = param
...         super().__init__(optimizer, last_epoch)
...
...     def get_lr(self):
...         # Custom logic here
...         return [base_lr * self.param for base_lr in self.base_lrs]
```

#### get_last_lr() → List[float]

Return last computed learning rate by current scheduler.

* **Returns:**
  List of last computed learning rates.

#### *abstractmethod* get_lr() → List[float]

Compute learning rate for each parameter group.

This method must be implemented by subclasses to define the
learning rate schedule logic.

* **Returns:**
  List of learning rates, one per parameter group.

#### load_state_dict(state_dict)

Load the scheduler state from a checkpoint.

* **Parameters:**
  **state_dict** – Scheduler state returned by state_dict().

#### state_dict()

Returns the state of the scheduler as a dict.

Contains all non-callable attributes that are specific to
the scheduler and required for checkpointing.

### olm.train.schedulers.base.abstractmethod(funcobj)

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
