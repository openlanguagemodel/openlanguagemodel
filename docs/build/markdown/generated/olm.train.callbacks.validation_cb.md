# olm.train.callbacks.validation_cb

Validation callback for running validation during training.

### Classes

| [`ValidationCallback`](#olm.train.callbacks.validation_cb.ValidationCallback)(val_dataloader[, ...])   | Callback to perform validation at specified intervals.   |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------|

### *class* olm.train.callbacks.validation_cb.TrainerCallback

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

### *class* olm.train.callbacks.validation_cb.ValidationCallback(val_dataloader, eval_every: int = 500, device: str = 'cuda', use_amp: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to perform validation at specified intervals.

* **Parameters:**
  * **val_dataloader** – Validation dataloader.
  * **eval_every** – Validate every N steps.
  * **device** – Device to run validation on.
  * **use_amp** – Whether to use automatic mixed precision.

#### on_step_end(trainer, step: int, loss: float) → None

Run validation after each optimization step if needed.
