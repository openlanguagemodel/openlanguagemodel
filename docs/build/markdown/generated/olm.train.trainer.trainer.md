# olm.train.trainer.trainer

### Classes

| [`Trainer`](#olm.train.trainer.trainer.Trainer)(model, optimizer, dataloader, ...)   | Manages the training loop for Open Language Model (OLM) architectures.   |
|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`TrainerCallback`](#olm.train.trainer.trainer.TrainerCallback)()                    | Base class for trainer callbacks.                                        |

### *class* olm.train.trainer.trainer.Any(\*args, \*\*kwargs)

Bases: `object`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### *class* olm.train.trainer.trainer.CrossEntropyLoss(\*args: [Any](#olm.train.trainer.trainer.Any), \*\*kwargs: [Any](#olm.train.trainer.trainer.Any))

Bases: [`LossBase`](#olm.train.trainer.trainer.LossBase)

#### forward(logits: torch.Tensor, y: torch.Tensor) → torch.Tensor

Apply loss to `logits` and `y`.

### *class* olm.train.trainer.trainer.DataLoader(\*args: [Any](#olm.train.trainer.trainer.Any), \*\*kwargs: [Any](#olm.train.trainer.trainer.Any))

Bases: `DataLoader`

Wrapper around PyTorch’s DataLoader with sensible defaults for LLM training.

This class extends torch.utils.data.DataLoader with:
- Better defaults for language model training
- Automatic worker configuration
- Pin memory optimization for GPU training
- Persistent workers for efficiency

* **Parameters:**
  * **dataset** – Dataset to load from (can be map-style or iterable).
  * **batch_size** – Number of samples per batch (default: 8).
  * **shuffle** – Whether to shuffle data at every epoch (default: False for iterable datasets).
  * **num_workers** – Number of worker processes for data loading (default: 0).
  * **pin_memory** – If True, tensors are copied to CUDA pinned memory (default: True).
  * **drop_last** – Drop the last incomplete batch if dataset size is not divisible by batch_size.
  * **persistent_workers** – Keep workers alive between epochs for faster startup (default: True if num_workers > 0).
  * **prefetch_factor** – Number of batches to prefetch per worker (default: 2).
  * **collate_fn** – Function to merge samples into batches.
  * **\*\*kwargs** – Additional arguments passed to torch.utils.data.DataLoader.

### Example

```pycon
>>> from olm.data.datasets import DataLoader
>>> loader = DataLoader(
...     dataset=my_dataset,
...     batch_size=16,
...     num_workers=4,
...     pin_memory=True,
... )
>>> for batch in loader:
...     # Training loop
...     pass
```

### *class* olm.train.trainer.trainer.LossBase(\*args: [Any](#olm.train.trainer.trainer.Any), \*\*kwargs: [Any](#olm.train.trainer.trainer.Any))

Bases: `Module`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Base class for all loss modules.

#### *abstractmethod* forward(logits: torch.Tensor, y: torch.Tensor, \*\*kwargs) → torch.Tensor

Apply loss to `logits` and `y`.

### *class* olm.train.trainer.trainer.TokenizerBase

Bases: [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all tokenizers in OLM.

Defines the interface for converting between text strings and integer token IDs.
Subclasses must implement encode and decode methods.

#### *abstractmethod* decode(tokens: torch.Tensor) → str

Converts a sequence of token IDs back into a text string.

* **Parameters:**
  **tokens** (*torch.Tensor*) – A 1D tensor or list of token IDs.
* **Returns:**
  The decoded text string.
* **Return type:**
  str

#### *abstractmethod* encode(text: str) → torch.Tensor

Converts a text string into a sequence of token IDs.

* **Parameters:**
  **text** (*str*) – The input text to tokenize.
* **Returns:**
  A 1D tensor containing the token IDs.
* **Return type:**
  torch.Tensor

### *class* olm.train.trainer.trainer.Trainer(model: torch.nn.Module, optimizer: torch.optim.Optimizer | ~typing.Type[torch.optim.Optimizer], dataloader: ~olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: ~typing.Type[~olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: ~typing.List[~olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: ~typing.Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True)

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
  [LossBase](#olm.train.trainer.trainer.LossBase)

#### callbacks

List of callbacks to execute during training.

* **Type:**
  List[[TrainerCallback](#olm.train.trainer.trainer.TrainerCallback)]

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

#### add_callback(callback: [TrainerCallback](#olm.train.trainer.trainer.TrainerCallback)) → None

Add a callback to the trainer.

#### remove_callback(callback: [TrainerCallback](#olm.train.trainer.trainer.TrainerCallback)) → None

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

### *class* olm.train.trainer.trainer.TrainerCallback

Bases: `object`

Base class for trainer callbacks.

#### on_batch_begin(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), batch_idx: int) → None

Called at the beginning of each batch.

#### on_batch_end(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), batch_idx: int, loss: float) → None

Called at the end of each batch.

#### on_epoch_begin(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the beginning of each epoch.

#### on_epoch_end(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), epoch: int) → None

Called at the end of each epoch.

#### on_step_begin(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), step: int) → None

Called at the beginning of each optimization step (after gradient accumulation).

#### on_step_end(trainer: [Trainer](#olm.train.trainer.trainer.Trainer), step: int, loss: float) → None

Called at the end of each optimization step.

#### on_train_begin(trainer: [Trainer](#olm.train.trainer.trainer.Trainer)) → None

Called at the beginning of training.

#### on_train_end(trainer: [Trainer](#olm.train.trainer.trainer.Trainer)) → None

Called at the end of training.

### *class* olm.train.trainer.trainer.Union

Bases: `object`

Represent a union type

E.g. for int | str

### *class* olm.train.trainer.trainer.WarmupCosineScheduler(\*args: [Any](#olm.train.trainer.trainer.Any), \*\*kwargs: [Any](#olm.train.trainer.trainer.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Combined warmup and cosine annealing scheduler.

Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
then applies cosine annealing decay to min_lr over the remaining steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **warmup_steps** – Number of warmup steps.
  * **total_steps** – Total number of training steps.
  * **min_lr** – Minimum learning rate after decay (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import WarmupCosineScheduler
>>> scheduler = WarmupCosineScheduler(
...     optimizer,
...     warmup_steps=1000,
...     total_steps=10000,
...     min_lr=1e-6
... )
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate with warmup and cosine decay.
