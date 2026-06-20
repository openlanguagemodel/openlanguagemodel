# `olm.train.schedulers.linear`

Source: [`src/olm/train/schedulers/linear.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L1)

Linear learning rate scheduler.

## Classes

### `LinearDecayLR(optimizer, total_steps: int, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/linear.py:66`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L66)

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `total_steps`: Total number of steps to decay over.
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import LinearDecayLR
scheduler = LinearDecayLR(optimizer, total_steps=1000)
for step in range(total_steps):
    train(...)
    scheduler.step()
```

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

**Parameters**

- `optimizer`: Wrapped optimizer.
- `total_steps`: Total number of steps for the schedule.
- `end_lr`: Target learning rate at the end (default: 0).
- `start_factor`: Initial learning rate multiplier (default: 1.0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import LinearLR
# Decay from initial LR to 0
scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
for step in range(total_steps):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/linear.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L42)

Compute learning rate using linear interpolation.
