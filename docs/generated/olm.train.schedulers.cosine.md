# `olm.train.schedulers.cosine`

Cosine annealing learning rate scheduler.

## Classes

### `CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1)`

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

- `get_lr(self)`
  Compute learning rate using cosine annealing.
