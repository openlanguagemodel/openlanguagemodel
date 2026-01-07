# olm.train

Training infrastructure for OLM.

### *class* olm.train.AdamW(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `AdamW`

AdamW optimizer with decoupled weight decay regularization.

This is a wrapper around PyTorch’s built-in AdamW implementation from
“Decoupled Weight Decay Regularization” (Loshchilov & Hutter, 2017).
Unlike the original Adam, weight decay is applied directly to the parameters
rather than being added to the gradient.

This implementation is commonly used for training large language models and
transformers, offering better generalization than standard Adam.

Note: This class inherits from PyTorch’s AdamW which ultimately inherits from
torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

* **Parameters:**
  * **params** – iterable of parameters to optimize or dicts defining parameter groups
  * **lr** – learning rate (default: 1e-3)
  * **betas** – coefficients used for computing running averages of gradient and its square
    (default: (0.9, 0.999))
  * **eps** – term added to the denominator to improve numerical stability (default: 1e-8)
  * **weight_decay** – weight decay coefficient (default: 0.01)
  * **amsgrad** – whether to use the AMSGrad variant (default: False)
  * **maximize** – maximize the params based on the objective, instead of minimizing (default: False)
  * **fused** – whether to use the fused implementation (default: None, auto-detect)

### Example

```pycon
>>> model = nn.Linear(10, 5)
>>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
>>> optimizer.zero_grad()
>>> loss = model(input).sum()
>>> loss.backward()
>>> optimizer.step()
```

### *class* olm.train.CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to save model checkpoints at specified intervals.

* **Parameters:**
  * **checkpoint_dir** – Directory to save checkpoints.
  * **save_every** – Save checkpoint every N steps.
  * **keep_last_n** – Keep only the last N checkpoints.
  * **save_best** – Whether to save the best model based on validation loss.

#### on_step_end(trainer, step: int, loss: float) → None

Save checkpoint after each optimization step if needed.

### *class* olm.train.CosineAnnealingLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **T_max** – Maximum number of iterations (steps).
  * **eta_min** – Minimum learning rate (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import CosineAnnealingLR
>>> scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
>>> for epoch in range(epochs):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using cosine annealing.

### *class* olm.train.EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to stop training early if validation loss doesn’t improve.

* **Parameters:**
  * **patience** – Number of validation checks to wait for improvement.
  * **min_delta** – Minimum change in validation loss to qualify as improvement.

#### on_step_end(trainer, step: int, loss: float) → None

Check for early stopping after each step.

### *class* olm.train.LRMonitorCallback(log_every: int = 100)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to monitor and log learning rate.

* **Parameters:**
  **log_every** – Log learning rate every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log learning rate after each optimization step if needed.

### *class* olm.train.LinearDecayLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **total_steps** – Total number of steps to decay over.
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import LinearDecayLR
>>> scheduler = LinearDecayLR(optimizer, total_steps=1000)
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using linear decay.

### *class* olm.train.LinearLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Linear learning rate scheduler.

Linearly decreases (or increases) the learning rate from the initial
learning rate to end_lr over total_steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **total_steps** – Total number of steps for the schedule.
  * **end_lr** – Target learning rate at the end (default: 0).
  * **start_factor** – Initial learning rate multiplier (default: 1.0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import LinearLR
>>> # Decay from initial LR to 0
>>> scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
>>> for step in range(total_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate using linear interpolation.

### *class* olm.train.Lion(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`OptimizerBase`](olm.train.optim.base.md#olm.train.optim.base.OptimizerBase)

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from “Symbolic Discovery of Optimization Algorithms”
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

* **Parameters:**
  * **params** – iterable of parameters to optimize or dicts defining parameter groups
  * **lr** – learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
  * **betas** – coefficients used for computing running averages (default: (0.9, 0.99))
  * **weight_decay** – weight decay coefficient (default: 0.0)
  * **use_triton** – whether to use Triton kernel for faster computation (default: False)

### Example

```pycon
>>> model = nn.Linear(10, 5)
>>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
>>> optimizer.zero_grad()
>>> loss = model(input).sum()
>>> loss.backward()
>>> optimizer.step()
```

#### step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized tensors to zero.

* **Parameters:**
  **set_to_none** – instead of setting to zero, set the grads to None.
  This is more memory efficient and can slightly improve performance.

### *class* olm.train.MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to log metrics to a JSONL file.

* **Parameters:**
  * **log_dir** – Directory to save logs.
  * **log_every** – Log metrics every N steps.

#### on_step_end(trainer, step: int, loss: float) → None

Log metrics after each optimization step if needed.

### *class* olm.train.OptimizerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Optimizer`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch’s Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### extra_repr() → str

String representation of the optimizer for debugging.

Override this in subclasses to provide useful information.

#### load_state_dict(state_dict: Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)])

Loads the optimizer state.

* **Parameters:**
  **state_dict** – optimizer state. Should be an object returned
  from a call to state_dict().

#### state_dict() → Dict[str, [Any](olm.train.trainer.trainer.md#olm.train.trainer.trainer.Any)]

Returns the state of the optimizer as a dict.

It contains two entries:

- `state`: dict holding current optimization state. Its content
  differs between optimizer classes.
- `param_groups`: list containing all parameter groups where each
  parameter group is a dict.

* **Returns:**
  Dictionary containing optimizer state

#### *abstractmethod* step(closure: Callable[[], float] | None = None) → float | None

Performs a single optimization step.

* **Parameters:**
  **closure** – A closure that reevaluates the model and returns the loss.
  Some optimization algorithms (e.g., L-BFGS) require multiple
  evaluations of the loss function.
* **Returns:**
  Optional loss value if closure is provided.

#### zero_grad(set_to_none: bool = True)

Sets gradients of all optimized tensors to zero or None.

* **Parameters:**
  **set_to_none** – Instead of setting to zero, set the grads to None.
  This is more memory efficient and can slightly improve performance.
  Default: True

### *class* olm.train.SchedulerBase(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `_LRScheduler`, [`ABC`](olm.train.schedulers.base.md#olm.train.schedulers.base.ABC)

Base class for all OLM learning rate schedulers.

This class extends PyTorch’s \_LRScheduler and provides a consistent
interface for implementing custom learning rate schedules. All OLM
schedulers should inherit from this class to maintain uniformity.

Subclasses must implement:
: - get_lr(): Compute the learning rate for the current step
  - \_get_closed_form_lr() (optional): Closed-form solution for efficiency

* **Parameters:**
  * **optimizer** – Wrapped PyTorch optimizer.
  * **last_epoch** – The index of the last epoch (default: -1).
  * **verbose** – If True, prints a message to stdout for each update (default: False).

### Example

```pycon
>>> class MyScheduler(SchedulerBase):
...     def __init__(self, optimizer, param, last_epoch=-1):
...         self.param = param
...         super().__init__(optimizer, last_epoch)
...
...     def get_lr(self):
...         # Custom logic here
...         return [base_lr * self.param for base_lr in self.base_lrs]
```

#### get_last_lr() → List[float]

Return last computed learning rate by current scheduler.

* **Returns:**
  List of last computed learning rates.

#### *abstractmethod* get_lr() → List[float]

Compute learning rate for each parameter group.

This method must be implemented by subclasses to define the
learning rate schedule logic.

* **Returns:**
  List of learning rates, one per parameter group.

#### load_state_dict(state_dict)

Load the scheduler state from a checkpoint.

* **Parameters:**
  **state_dict** – Scheduler state returned by state_dict().

#### state_dict()

Returns the state of the scheduler as a dict.

Contains all non-callable attributes that are specific to
the scheduler and required for checkpointing.

### *class* olm.train.ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)

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

### *class* olm.train.Trainer(model: torch.nn.Module, optimizer: torch.optim.Optimizer | ~typing.Type[torch.optim.Optimizer], dataloader: ~olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: ~typing.Type[~olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: ~typing.List[~olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: ~typing.Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True)

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
  List[[TrainerCallback](#olm.train.TrainerCallback)]

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

### *class* olm.train.TrainerCallback

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

### *class* olm.train.ValidationCallback(val_dataloader, eval_every: int = 500, device: str = 'cuda', use_amp: bool = True)

Bases: [`TrainerCallback`](olm.train.trainer.trainer.md#olm.train.trainer.trainer.TrainerCallback)

Callback to perform validation at specified intervals.

* **Parameters:**
  * **val_dataloader** – Validation dataloader.
  * **eval_every** – Validate every N steps.
  * **device** – Device to run validation on.
  * **use_amp** – Whether to use automatic mixed precision.

#### on_step_end(trainer, step: int, loss: float) → None

Run validation after each optimization step if needed.

### *class* olm.train.WarmupCosineScheduler(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### *class* olm.train.WarmupLR(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`SchedulerBase`](olm.train.schedulers.base.md#olm.train.schedulers.base.SchedulerBase)

Learning rate warmup scheduler.

Linearly increases the learning rate from 0 to the base learning rate
over warmup_steps.

* **Parameters:**
  * **optimizer** – Wrapped optimizer.
  * **warmup_steps** – Number of warmup steps.
  * **start_lr** – Initial learning rate (default: 0).
  * **last_epoch** – The index of last epoch (default: -1).

### Example

```pycon
>>> from olm.train.schedulers import WarmupLR
>>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
>>> for step in range(warmup_steps):
...     train(...)
...     scheduler.step()
```

#### get_lr()

Compute learning rate during warmup.

### Modules

| [`callbacks`](olm.train.callbacks.md#module-olm.train.callbacks)                | Callbacks for the Trainer class.           |
|---------------------------------------------------------------------------------|--------------------------------------------|
| [`optim`](olm.train.optim.md#module-olm.train.optim)                            |                                            |
| [`regularization`](olm.train.regularization.md#module-olm.train.regularization) |                                            |
| [`schedulers`](olm.train.schedulers.md#module-olm.train.schedulers)             | Learning rate schedulers for OLM training. |
| [`trainer`](olm.train.trainer.md#module-olm.train.trainer)                      |                                            |
