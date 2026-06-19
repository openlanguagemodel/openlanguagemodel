# `olm.train.callbacks.throughput_cb`

Throughput monitoring callback.

## Classes

### `ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)`

Callback to monitor training throughput (tokens/sec, samples/sec).

Args:
    log_every: Log throughput every N steps.
    context_length: Length of each sequence.
    batch_size: Total batch size (including gradient accumulation).

#### Methods

- `on_step_begin(self, trainer, step: int) -> None`
  Record start time of the step.
- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Calculate and log throughput.
