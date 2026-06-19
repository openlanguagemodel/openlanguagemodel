# `olm.train.schedulers.base`

Source: [`src/olm/train/schedulers/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L1)

Base learning rate scheduler for OLM.

## Classes

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
