# `olm.train.trainer.ddp_trainer`

Source: [`src/olm/train/trainer/ddp_trainer.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/ddp_trainer.py#L1)

Distributed Data Parallel (DDP) Trainer using PyTorch's native DDP.

## Classes

### `DDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, ddp_backend: str | None = None, find_unused_parameters: bool = False, broadcast_buffers: bool = True, bucket_cap_mb: int = 25, gradient_as_bucket_view: bool = True, static_graph: bool = False)`

**Bases:** `olm.train.trainer.trainer.Trainer`

Source: [`src/olm/train/trainer/ddp_trainer.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/ddp_trainer.py#L27)

Trainer with PyTorch Distributed Data Parallel (DDP) support.

Wraps the model with torch.nn.parallel.DistributedDataParallel and handles:
- Distributed sampler setup
- Gradient synchronization (with no_sync for gradient accumulation)
- Metrics aggregation across ranks
- Checkpoint saving on rank 0

Args:
    model: Model to train.
    optimizer: Optimizer instance or class.
    dataloader: DataLoader (will add DistributedSampler if needed).
    device: Device for training.
    context_length: Max sequence length.
    grad_accum_steps: Gradient accumulation steps.
    use_amp: Use automatic mixed precision.
    loss: Loss function class.
    callbacks: Training callbacks.
    scheduler: Learning rate scheduler.
    grad_clip_norm: Gradient clipping threshold.
    warmup_steps: Warmup steps for scheduler.
    total_steps: Total training steps.
    min_lr: Minimum learning rate.
    learning_rate: Learning rate (if optimizer is a class).
    weight_decay: Weight decay (if optimizer is a class).
    use_warmup_cosine: Use warmup+cosine scheduler by default.
    ddp_backend: DDP backend ('nccl' for GPU, 'gloo' for CPU, None for auto).
    find_unused_parameters: DDP parameter for models with unused params.
    broadcast_buffers: Broadcast model buffers at beginning of forward.
    bucket_cap_mb: DDP bucket size in MB for gradient communication.
    gradient_as_bucket_view: Use gradient views to reduce memory (recommended).
    static_graph: Set to True if model graph doesn't change (optimization).

Example:
    >>> # Launch with: torchrun --nproc_per_node=4 train.py
    >>> from olm.core.dist import setup_distributed
    >>> setup_distributed()
    >>>
    >>> trainer = DDPTrainer(
    ...     model=model,
    ...     optimizer=torch.optim.AdamW,
    ...     dataloader=dataloader,
    ...     device=f"cuda:{get_local_rank()}",
    ...     context_length=512,
    ...     learning_rate=3e-4
    ... )
    >>> trainer.train(epochs=10)

#### Methods

##### `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`

Source: [`src/olm/train/trainer/ddp_trainer.py:150`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/ddp_trainer.py#L150)

Training loop with DDP support.

Args:
    epochs: Number of epochs.
    log_interval: Log every N steps.
    max_steps: Maximum steps to train.
    steps_per_epoch: Max steps per epoch.

Returns:
    List of loss values (only on rank 0).
