# olm.train.callbacks.throughput_cb

Throughput monitoring callback.

### Classes

| [`ThroughputCallback`](#olm.train.callbacks.throughput_cb.ThroughputCallback)([log_every, ...])   | Callback to monitor training throughput (tokens/sec, samples/sec).   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|

### *class* olm.train.callbacks.throughput_cb.ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)

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

### *class* olm.train.callbacks.throughput_cb.TrainerCallback

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
