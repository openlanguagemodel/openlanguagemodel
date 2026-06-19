# `olm.train.schedulers.base`

Base learning rate scheduler for OLM.

## Classes

### `SchedulerBase(optimizer, last_epoch: int = -1, verbose: bool = False)`

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

- `get_last_lr(self) -> List[float]`
  Return last computed learning rate by current scheduler.
- `get_lr(self) -> List[float]`
  Compute learning rate for each parameter group.
- `load_state_dict(self, state_dict)`
  Load the scheduler state from a checkpoint.
- `state_dict(self)`
  Returns the state of the scheduler as a dict.
