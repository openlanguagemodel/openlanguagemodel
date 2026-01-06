# olm.train.callbacks

Callbacks for the Trainer class.

### *class* olm.train.callbacks.CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to save model checkpoints at specified intervals.

* **Parameters:**
  * **checkpoint_dir** – Directory to save checkpoints.
  * **save_every** – Save checkpoint every N steps.
  * **keep_last_n** – Keep only the last N checkpoints.
  * **save_best** – Whether to save the best model based on validation loss.

#### on_step_end(trainer, step: int, loss: float) → None

Save checkpoint after each optimization step if needed.

### *class* olm.train.callbacks.EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to stop training early if validation loss doesn’t improve.

* **Parameters:**
  * **patience** – Number of validation checks to wait for improvement.
  * **min_delta** – Minimum change in validation loss to qualify as improvement.

#### on_step_end(trainer, step: int, loss: float) → None

Check for early stopping after each step.

### *class* olm.train.callbacks.LRMonitorCallback(log_every: int = 100)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to monitor and log learning rate.

* **Parameters:**
  **log_every** – Log learning rate every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log learning rate after each optimization step if needed.

### *class* olm.train.callbacks.MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to log metrics to a JSONL file.

* **Parameters:**
  * **log_dir** – Directory to save logs.
  * **log_every** – Log metrics every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log metrics after each optimization step if needed.

### *class* olm.train.callbacks.ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to monitor training throughput (tokens/sec, samples/sec).

* **Parameters:**
  * **log_every** – Log throughput every N steps.
  * **context_length** – Length of each sequence.
  * **batch_size** – Total batch size (including gradient accumulation).

#### on_step_begin(trainer, step: int) → None

Record start time of the step.

#### on_step_end(trainer, step: int, loss: float) → None

Calculate and log throughput.

### *class* olm.train.callbacks.ValidationCallback(val_dataloader, eval_every: int = 500, device: str = 'cuda', use_amp: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to perform validation at specified intervals.

* **Parameters:**
  * **val_dataloader** – Validation dataloader.
  * **eval_every** – Validate every N steps.
  * **device** – Device to run validation on.
  * **use_amp** – Whether to use automatic mixed precision.

#### on_step_end(trainer, step: int, loss: float) → None

Run validation after each optimization step if needed.

### Modules

| [`checkpoint_cb`](olm.train.callbacks.checkpoint_cb.md#module-olm.train.callbacks.checkpoint_cb)             | Checkpoint callback for saving model checkpoints during training.   |
|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`early_stopping_cb`](olm.train.callbacks.early_stopping_cb.md#module-olm.train.callbacks.early_stopping_cb) | Early stopping callback to prevent overfitting.                     |
| [`lr_monitor_cb`](olm.train.callbacks.lr_monitor_cb.md#module-olm.train.callbacks.lr_monitor_cb)             | Learning rate monitoring callback.                                  |
| [`metrics_logger_cb`](olm.train.callbacks.metrics_logger_cb.md#module-olm.train.callbacks.metrics_logger_cb) | Metrics logging callback for tracking training metrics.             |
| [`throughput_cb`](olm.train.callbacks.throughput_cb.md#module-olm.train.callbacks.throughput_cb)             | Throughput monitoring callback.                                     |
| [`validation_cb`](olm.train.callbacks.validation_cb.md#module-olm.train.callbacks.validation_cb)             | Validation callback for running validation during training.         |
