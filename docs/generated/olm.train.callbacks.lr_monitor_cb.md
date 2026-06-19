# `olm.train.callbacks.lr_monitor_cb`

Learning rate monitoring callback.

## Classes

### `LRMonitorCallback(log_every: int = 100)`

Callback to monitor and log learning rate.

Args:
    log_every: Log learning rate every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log learning rate after each optimization step if needed.
