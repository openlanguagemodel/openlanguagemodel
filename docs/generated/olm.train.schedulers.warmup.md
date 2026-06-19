# `olm.train.schedulers.warmup`

Source: [`src/olm/train/schedulers/warmup.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L1)

Warmup learning rate scheduler.

## Classes

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
