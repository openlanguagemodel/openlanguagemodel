# `olm.train.callbacks`

Callbacks for the Trainer class.

## Classes

### `CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)`

Callback to save model checkpoints at specified intervals.

Args:
    checkpoint_dir: Directory to save checkpoints.
    save_every: Save checkpoint every N steps.
    keep_last_n: Keep only the last N checkpoints.
    save_best: Whether to save the best model based on validation loss.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Save checkpoint after each optimization step if needed.

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

Callback to stop training early if validation loss doesn't improve.

Args:
    patience: Number of validation checks to wait for improvement.
    min_delta: Minimum change in validation loss to qualify as improvement.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Check for early stopping after each step.

### `LRMonitorCallback(log_every: int = 100)`

Callback to monitor and log learning rate.

Args:
    log_every: Log learning rate every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log learning rate after each optimization step if needed.

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

Callback to log metrics to a JSONL file.

Args:
    log_dir: Directory to save logs.
    log_every: Log metrics every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log metrics after each optimization step if needed.

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

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

Callback to perform validation at specified intervals.

Args:
    val_dataloader: Validation dataloader.
    eval_every: Validate every N steps.
    device: Device to run validation on.
    use_amp: Whether to use automatic mixed precision.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Run validation after each optimization step if needed.
