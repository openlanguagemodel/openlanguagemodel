# `olm.train.schedulers.linear`

Linear learning rate scheduler.

## Classes

### `LinearDecayLR(optimizer, total_steps: int, last_epoch: int = -1)`

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

- `get_lr(self)`
  Compute learning rate using linear decay.

### `LinearLR(optimizer, total_steps: int, end_lr: float = 0, start_factor: float = 1.0, last_epoch: int = -1)`

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

- `get_lr(self)`
  Compute learning rate using linear interpolation.
