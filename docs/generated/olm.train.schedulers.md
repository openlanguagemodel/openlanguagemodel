# olm.train.schedulers

Learning rate schedulers for OLM training.

### *class* olm.train.schedulers.CosineAnnealingLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.train.schedulers.LinearDecayLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **total_steps** – Total number of steps to decay over.
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import LinearDecayLR
>>> scheduler = LinearDecayLR(optimizer, total_steps=1000)
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using linear decay.

### *class* olm.train.schedulers.LinearLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Linear learning rate scheduler.

Linearly decreases (or increases) the learning rate from the initial
learning rate to end_lr over total_steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **total_steps** – Total number of steps for the schedule.
  * **end_lr** – Target learning rate at the end (default: 0).
  * **start_factor** – Initial learning rate multiplier (default: 1.0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import LinearLR
>>> # Decay from initial LR to 0
>>> scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using linear interpolation.

### *class* olm.train.schedulers.SchedulerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.train.schedulers.WarmupCosineScheduler(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Combined warmup and cosine annealing scheduler.

Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
then applies cosine annealing decay to min_lr over the remaining steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **warmup_steps** – Number of warmup steps.
  * **total_steps** – Total number of training steps.
  * **min_lr** – Minimum learning rate after decay (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import WarmupCosineScheduler
>>> scheduler = WarmupCosineScheduler(
...     optimizer,
...     warmup_steps=1000,
...     total_steps=10000,
...     min_lr=1e-6
... )
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate with warmup and cosine decay.

### *class* olm.train.schedulers.WarmupLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Learning rate warmup scheduler.

Linearly increases the learning rate from 0 to the base learning rate
over warmup_steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **warmup_steps** – Number of warmup steps.
  * **start_lr** – Initial learning rate (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import WarmupLR
>>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
>>> for step in range(warmup_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate during warmup.

### Modules

| [`base`](olm.train.schedulers.base.md#module-olm.train.schedulers.base)       | Base learning rate scheduler for OLM.     |
|-------------------------------------------------------------------------------|-------------------------------------------|
| [`cosine`](olm.train.schedulers.cosine.md#module-olm.train.schedulers.cosine) | Cosine annealing learning rate scheduler. |
| [`linear`](olm.train.schedulers.linear.md#module-olm.train.schedulers.linear) | Linear learning rate scheduler.           |
| [`warmup`](olm.train.schedulers.warmup.md#module-olm.train.schedulers.warmup) | Warmup learning rate scheduler.           |
