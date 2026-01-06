# olm.train.callbacks.early_stopping_cb

Early stopping callback to prevent overfitting.

### Classes

| [`EarlyStoppingCallback`](#olm.train.callbacks.early_stopping_cb.EarlyStoppingCallback)([patience, min_delta])   | Callback to stop training early if validation loss doesn't improve.   |
|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|

### *class* olm.train.callbacks.early_stopping_cb.EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to stop training early if validation loss doesn’t improve.

* **Parameters:**
  * **patience** – Number of validation checks to wait for improvement.
  * **min_delta** – Minimum change in validation loss to qualify as improvement.

#### on_step_end(trainer, step: int, loss: float) → None

Check for early stopping after each step.

### *class* olm.train.callbacks.early_stopping_cb.TrainerCallback

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
