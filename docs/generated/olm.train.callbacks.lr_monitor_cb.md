# `olm.train.callbacks.lr_monitor_cb`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L1)

Learning rate monitoring callback.

## Classes

### `LRMonitorCallback(log_every: int = 100)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L8)

Callback to monitor and log learning rate.

Args:
    log_every: Log learning rate every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:19`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L19)

Log learning rate after each optimization step if needed.
