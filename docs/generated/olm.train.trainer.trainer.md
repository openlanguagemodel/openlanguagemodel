# `olm.train.trainer.trainer`

Source: [`src/olm/train/trainer/trainer.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L1)

## Classes

### `Trainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True)`

Source: [`src/olm/train/trainer/trainer.py:53`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L53)

Manages the training loop for Open Language Model (OLM) architectures.

This trainer handles the core training logic including:
- Automatic Mixed Precision (AMP) scaling
- Gradient accumulation
- Device management (moving data/models to GPU)
- Optimization steps
- Callbacks for validation, checkpointing, and custom logic
- Learning rate scheduling support
- Gradient clipping

Training contract:
    The dataloader must yield ``(input_ids, labels)``. Both tensors are
    moved to ``device`` and truncated to ``context_length``. The model must
    return logits shaped ``[batch, seq_len, vocab_size]`` so the configured
    loss can compare logits against ``labels`` shaped ``[batch, seq_len]``.

Attributes:
    model (Pipeline): The model to train.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    dataloader (olm.data.datasets.data_loader.DataLoader): The data provider.
    device (str): The device to train on (e.g., 'cuda', 'cpu').
    context_length (int): The maximum sequence length for training.
    grad_accum_steps (int): Number of steps to accumulate gradients before updating.
    use_amp (bool): Whether to use Automatic Mixed Precision.
    scaler (GradScaler): Gradient scaler for AMP.
    loss (LossBase): The loss function instance.
    callbacks (List[TrainerCallback]): List of callbacks to execute during training.
    scheduler (Optional): Learning rate scheduler to step after each optimization step.
    global_step (int): Current global step count.
    current_epoch (int): Current epoch number.
    total_tokens_processed (int): Total number of tokens processed during training.
    step_start_time (float): Timestamp of the current step start.
    training_start_time (float): Timestamp when training began.

#### Methods

##### `add_callback(self, callback: olm.train.trainer.trainer.TrainerCallback) -> None`

Source: [`src/olm/train/trainer/trainer.py:213`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L213)

Add a callback to the trainer.

##### `remove_callback(self, callback: olm.train.trainer.trainer.TrainerCallback) -> None`

Source: [`src/olm/train/trainer/trainer.py:217`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L217)

Remove a callback from the trainer.

##### `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`

Source: [`src/olm/train/trainer/trainer.py:267`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L267)

Executes the training loop for a specified number of epochs.

Args:
    epochs (int): The number of complete passes through the dataset.
    log_interval (int): How often to print the loss. Defaults to 10.
    max_steps (int, optional): Maximum number of steps to train for.
    steps_per_epoch (int, optional): Maximum number of steps per epoch. Defaults to None (unlimited).

Returns:
    list[float]: A list of recorded loss values.

### `TrainerCallback()`

Source: [`src/olm/train/trainer/trainer.py:17`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L17)

Base class for trainer callbacks.

#### Methods

##### `on_batch_begin(self, trainer: 'Trainer', batch_idx: int) -> None`

Source: [`src/olm/train/trainer/trainer.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L36)

Called at the beginning of each batch.

##### `on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None`

Source: [`src/olm/train/trainer/trainer.py:40`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L40)

Called at the end of each batch.

##### `on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None`

Source: [`src/olm/train/trainer/trainer.py:28`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L28)

Called at the beginning of each epoch.

##### `on_epoch_end(self, trainer: 'Trainer', epoch: int) -> None`

Source: [`src/olm/train/trainer/trainer.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L32)

Called at the end of each epoch.

##### `on_step_begin(self, trainer: 'Trainer', step: int) -> None`

Source: [`src/olm/train/trainer/trainer.py:44`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L44)

Called at the beginning of each optimization step (after gradient accumulation).

##### `on_step_end(self, trainer: 'Trainer', step: int, loss: float) -> None`

Source: [`src/olm/train/trainer/trainer.py:48`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L48)

Called at the end of each optimization step.

##### `on_train_begin(self, trainer: 'Trainer') -> None`

Source: [`src/olm/train/trainer/trainer.py:20`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L20)

Called at the beginning of training.

##### `on_train_end(self, trainer: 'Trainer') -> None`

Source: [`src/olm/train/trainer/trainer.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/trainer.py#L24)

Called at the end of training.
