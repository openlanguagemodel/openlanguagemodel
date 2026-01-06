# olm.train.trainer

### *class* olm.train.trainer.CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to save model checkpoints at specified intervals.

* **Parameters:**
  * **checkpoint_dir** – Directory to save checkpoints.
  * **save_every** – Save checkpoint every N steps.
  * **keep_last_n** – Keep only the last N checkpoints.
  * **save_best** – Whether to save the best model based on validation loss.

#### on_step_end(trainer, step: int, loss: float) → None

Save checkpoint after each optimization step if needed.

### *class* olm.train.trainer.EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to stop training early if validation loss doesn’t improve.

* **Parameters:**
  * **patience** – Number of validation checks to wait for improvement.
  * **min_delta** – Minimum change in validation loss to qualify as improvement.

#### on_step_end(trainer, step: int, loss: float) → None

Check for early stopping after each step.

### *class* olm.train.trainer.LRMonitorCallback(log_every: int = 100)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to monitor and log learning rate.

* **Parameters:**
  **log_every** – Log learning rate every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log learning rate after each optimization step if needed.

### *class* olm.train.trainer.MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to log metrics to a JSONL file.

* **Parameters:**
  * **log_dir** – Directory to save logs.
  * **log_every** – Log metrics every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log metrics after each optimization step if needed.

### *class* olm.train.trainer.ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)

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

### *class* olm.train.trainer.Trainer(model: torch.nn.Module, optimizer: torch.optim.Optimizer | ~typing.Type[torch.optim.Optimizer], dataloader: ~olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: ~typing.Type[~olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: ~typing.List[~olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: ~typing.Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True)

Bases: `object`

Manages the training loop for Open Language Model (OLM) architectures.

This trainer handles the core training logic including:
- Automatic Mixed Precision (AMP) scaling
- Gradient accumulation
- Device management (moving data/models to GPU)
- Optimization steps
- Callbacks for validation, checkpointing, and custom logic
- Learning rate scheduling support
- Gradient clipping

#### model

The model to train.

* **Type:**
  Pipeline

#### optimizer

The optimizer to use.

* **Type:**
  torch.optim.Optimizer

#### dataloader

The data provider.

* **Type:**
  [olm.data.datasets.data_loader.DataLoader](olm.data.datasets.data_loader.md#olm.data.datasets.data_loader.DataLoader)

#### device

The device to train on (e.g., ‘cuda’, ‘cpu’).

* **Type:**
  str

#### context_length

The maximum sequence length for training.

* **Type:**
  int

#### grad_accum_steps

Number of steps to accumulate gradients before updating.

* **Type:**
  int

#### use_amp

Whether to use Automatic Mixed Precision.

* **Type:**
  bool

#### scaler

Gradient scaler for AMP.

* **Type:**
  GradScaler

#### loss

The loss function instance.

* **Type:**
  [LossBase](olm.train.trainer.trainer.md#olm.train.trainer.trainer.LossBase)

#### callbacks

List of callbacks to execute during training.

* **Type:**
  List[[TrainerCallback](#olm.train.trainer.TrainerCallback)]

#### scheduler

Learning rate scheduler to step after each optimization step.

* **Type:**
  Optional

#### global_step

Current global step count.

* **Type:**
  int

#### current_epoch

Current epoch number.

* **Type:**
  int

#### add_callback(callback: [TrainerCallback](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)) → None

Add a callback to the trainer.

#### remove_callback(callback: [TrainerCallback](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)) → None

Remove a callback from the trainer.

#### train(epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) → list[float]

Executes the training loop for a specified number of epochs.

* **Parameters:**
  * **epochs** (*int*) – The number of complete passes through the dataset.
  * **log_interval** (*int*) – How often to print the loss. Defaults to 10.
  * **max_steps** (*int* *,* *optional*) – Maximum number of steps to train for.
  * **steps_per_epoch** (*int* *,* *optional*) – Maximum number of steps per epoch. Defaults to None (unlimited).
* **Returns:**
  A list of recorded loss values.
* **Return type:**
  list[float]

### *class* olm.train.trainer.TrainerCallback

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

### *class* olm.train.trainer.ValidationCallback(val_dataloader, eval_every: int = 500, device: str = 'cuda', use_amp: bool = True)

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

| [`trainer`](olm.train.trainer.trainer.md#module-olm.train.trainer.trainer)   |    |
|------------------------------------------------------------------------------|----|
