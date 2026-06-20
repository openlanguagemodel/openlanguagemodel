# `olm.train.callbacks`

Source: [`src/olm/train/callbacks/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/__init__.py#L1)

Callbacks for the Trainer class.

## Classes

### `CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L12)

Callback to save model checkpoints at specified intervals.

**Parameters**

- `checkpoint_dir`: Directory to save checkpoints.
- `save_every`: Save checkpoint every N steps.
- `keep_last_n`: Keep only the last N checkpoints.
- `save_best`: Whether to save the best model based on validation loss.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L36)

Save checkpoint after each optimization step if needed.

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L8)

Callback to stop training early if validation loss doesn't improve.

**Parameters**

- `patience`: Number of validation checks to wait for improvement.
- `min_delta`: Minimum change in validation loss to qualify as improvement.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L24)

Check for early stopping after each step.

### `LRMonitorCallback(log_every: int = 100)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L8)

Callback to monitor and log learning rate.

**Parameters**

- `log_every`: Log learning rate every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:19`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L19)

Log learning rate after each optimization step if needed.

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L10)

Callback to log metrics to a JSONL file.

**Parameters**

- `log_dir`: Directory to save logs.
- `log_every`: Log metrics every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L30)

Log metrics after each optimization step if needed.

### `ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/throughput_cb.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L9)

Callback to monitor training throughput (tokens/sec, samples/sec).

**Parameters**

- `log_every`: Log throughput every N steps.
- `context_length`: Length of each sequence.
- `batch_size`: Total batch size (including gradient accumulation).

#### Methods

##### `on_step_begin(self, trainer, step: int) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L30)

Record start time of the step.

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L34)

Calculate and log throughput.

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/validation_cb.py:11`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L11)

Callback to perform validation at specified intervals.

**Parameters**

- `val_dataloader`: Validation dataloader.
- `eval_every`: Validate every N steps.
- `device`: Device to run validation on.
- `use_amp`: Whether to use automatic mixed precision.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/validation_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L36)

Run validation after each optimization step if needed.
