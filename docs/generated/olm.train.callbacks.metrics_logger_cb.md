# `olm.train.callbacks.metrics_logger_cb`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L1)

Metrics logging callback for tracking training metrics.

## Classes

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L10)

Callback to log metrics to a JSONL file.

Args:
    log_dir: Directory to save logs.
    log_every: Log metrics every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L30)

Log metrics after each optimization step if needed.
