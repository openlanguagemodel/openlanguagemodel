# `olm.train.schedulers.cosine`

Source: [`src/olm/train/schedulers/cosine.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L1)

Cosine annealing learning rate scheduler.

## Classes

### `CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/cosine.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L7)

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `T_max`: Maximum number of iterations (steps).
- `eta_min`: Minimum learning rate (default: 0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/cosine.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L39)

Compute learning rate using cosine annealing.
