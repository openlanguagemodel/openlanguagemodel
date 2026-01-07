# olm.train.schedulers.cosine

Cosine annealing learning rate scheduler.

### Classes

| [`CosineAnnealingLR`](#olm.train.schedulers.cosine.CosineAnnealingLR)(\*args, \*\*kwargs)   | Cosine annealing learning rate scheduler.   |
|---------------------------------------------------------------------------------------------|---------------------------------------------|

### *class* olm.train.schedulers.cosine.CosineAnnealingLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **T_max** – Maximum number of iterations (steps).
  * **eta_min** – Minimum learning rate (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import CosineAnnealingLR
>>> scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
>>> for epoch in range(epochs):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using cosine annealing.

### *class* olm.train.schedulers.cosine.SchedulerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `_LRScheduler`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

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
