# olm.train.callbacks.lr_monitor_cb

Learning rate monitoring callback.

### Classes

| [`LRMonitorCallback`](#olm.train.callbacks.lr_monitor_cb.LRMonitorCallback)([log_every])   | Callback to monitor and log learning rate.   |
|--------------------------------------------------------------------------------------------|----------------------------------------------|

### *class* olm.train.callbacks.lr_monitor_cb.LRMonitorCallback(log_every: int = 100)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to monitor and log learning rate.

* **Parameters:**
  **log_every** – Log learning rate every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log learning rate after each optimization step if needed.

### *class* olm.train.callbacks.lr_monitor_cb.TrainerCallback

Bases: `object`

Base class for trainer callbacks.

#### on_batch_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), batch_idx: int) → None

Called at the beginning of each batch.

#### on_batch_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), batch_idx: int, loss: float) → None

Called at the end of each batch.

#### on_epoch_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the beginning of each epoch.

#### on_epoch_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the end of each epoch.

#### on_step_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), step: int) → None

Called at the beginning of each optimization step (after gradient accumulation).

#### on_step_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer), step: int, loss: float) → None

Called at the end of each optimization step.

#### on_train_begin(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer)) → None

Called at the beginning of training.

#### on_train_end(trainer: [Trainer](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Trainer)) → None

Called at the end of training.
