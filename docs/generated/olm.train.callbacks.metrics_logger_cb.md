# `olm.train.callbacks.metrics_logger_cb`

Metrics logging callback for tracking training metrics.

## Classes

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

Callback to log metrics to a JSONL file.

Args:
    log_dir: Directory to save logs.
    log_every: Log metrics every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log metrics after each optimization step if needed.
