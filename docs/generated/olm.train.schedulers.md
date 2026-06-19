# `olm.train.schedulers`

Source: [`src/olm/train/schedulers/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/__init__.py#L1)

Learning rate schedulers for OLM training.

## Classes

### `CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/cosine.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L7)

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

Args:
    optimizer: Wrapped optimizer.
    T_max: Maximum number of iterations (steps).
    eta_min: Minimum learning rate (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import CosineAnnealingLR
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    >>> for epoch in range(epochs):
    ...     train(...)
    ...     scheduler.step()

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/cosine.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L39)

Compute learning rate using cosine annealing.

### `LinearDecayLR(optimizer, total_steps: int, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/linear.py:66`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L66)

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

Args:
    optimizer: Wrapped optimizer.
    total_steps: Total number of steps to decay over.
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import LinearDecayLR
    >>> scheduler = LinearDecayLR(optimizer, total_steps=1000)
    >>> for step in range(total_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/linear.py:89`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L89)

Compute learning rate using linear decay.

### `LinearLR(optimizer, total_steps: int, end_lr: float = 0, start_factor: float = 1.0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/linear.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L6)

Linear learning rate scheduler.

Linearly decreases (or increases) the learning rate from the initial
learning rate to end_lr over total_steps.

Args:
    optimizer: Wrapped optimizer.
    total_steps: Total number of steps for the schedule.
    end_lr: Target learning rate at the end (default: 0).
    start_factor: Initial learning rate multiplier (default: 1.0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import LinearLR
    >>> # Decay from initial LR to 0
    >>> scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
    >>> for step in range(total_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/linear.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L42)

Compute learning rate using linear interpolation.

### `SchedulerBase(optimizer, last_epoch: int = -1, verbose: bool = False)`

**Bases:** `_LRScheduler`, `ABC`

Source: [`src/olm/train/schedulers/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L8)

Base class for all OLM learning rate schedulers.

This class extends PyTorch's _LRScheduler and provides a consistent
interface for implementing custom learning rate schedules. All OLM
schedulers should inherit from this class to maintain uniformity.

Subclasses must implement:
    - get_lr(): Compute the learning rate for the current step
    - _get_closed_form_lr() (optional): Closed-form solution for efficiency

Args:
    optimizer: Wrapped PyTorch optimizer.
    last_epoch: The index of the last epoch (default: -1).
    verbose: If True, prints a message to stdout for each update (default: False).

Example:
    >>> class MyScheduler(SchedulerBase):
    ...     def __init__(self, optimizer, param, last_epoch=-1):
    ...         self.param = param
    ...         super().__init__(optimizer, last_epoch)
    ...
    ...     def get_lr(self):
    ...         # Custom logic here
    ...         return [base_lr * self.param for base_lr in self.base_lrs]

#### Methods

##### `get_last_lr(self) -> List[float]`

Source: [`src/olm/train/schedulers/base.py:64`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L64)

Return last computed learning rate by current scheduler.

Returns:
    List of last computed learning rates.

##### `get_lr(self) -> List[float]`

Source: [`src/olm/train/schedulers/base.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L39)

Compute learning rate for each parameter group.

This method must be implemented by subclasses to define the
learning rate schedule logic.

Returns:
    List of learning rates, one per parameter group.

##### `load_state_dict(self, state_dict)`

Source: [`src/olm/train/schedulers/base.py:86`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L86)

Load the scheduler state from a checkpoint.

Args:
    state_dict: Scheduler state returned by state_dict().

##### `state_dict(self)`

Source: [`src/olm/train/schedulers/base.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L73)

Returns the state of the scheduler as a dict.

Contains all non-callable attributes that are specific to
the scheduler and required for checkpointing.

### `WarmupCosineScheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/warmup.py:62`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L62)

Combined warmup and cosine annealing scheduler.

Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
then applies cosine annealing decay to min_lr over the remaining steps.

Args:
    optimizer: Wrapped optimizer.
    warmup_steps: Number of warmup steps.
    total_steps: Total number of training steps.
    min_lr: Minimum learning rate after decay (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
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

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/warmup.py:102`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L102)

Compute learning rate with warmup and cosine decay.

### `WarmupLR(optimizer, warmup_steps: int, start_lr: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/warmup.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L6)

Learning rate warmup scheduler.

Linearly increases the learning rate from 0 to the base learning rate
over warmup_steps.

Args:
    optimizer: Wrapped optimizer.
    warmup_steps: Number of warmup steps.
    start_lr: Initial learning rate (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import WarmupLR
    >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
    >>> for step in range(warmup_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/warmup.py:38`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L38)

Compute learning rate during warmup.
