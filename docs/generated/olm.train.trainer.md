# `olm.train.trainer`

## Functions

### `AutoTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str | olm.train.device.DeviceConfig = 'auto', context_length: int = 1024, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None, verbose: bool = True, ddp_find_unused_parameters: bool = False, ddp_broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, fsdp_min_num_params: int = 100000000, fsdp_transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, fsdp_backward_prefetch: str = 'BACKWARD_PRE', fsdp_limit_all_gathers: bool = True, fsdp_use_orig_params: bool = True) -> olm.train.trainer.trainer.Trainer | olm.train.trainer.ddp_trainer.DDPTrainer | olm.train.trainer.fsdp_trainer.FSDPTrainer`

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles all
the complexity of distributed training setup, device selection, and
parameter configuration.

Args:
    model: Model to train.
    optimizer: Optimizer instance or class.
    dataloader: DataLoader for training data.
    device: Device configuration. Options:
        - "auto": Full auto-detection (recommended)
        - "cuda:auto": Force CUDA with auto-configuration
        - "cpu:auto": Force CPU with auto-configuration
        - "cuda", "cuda:0", "cpu": Legacy device strings
        - DeviceConfig object: Custom configuration
    context_length: Maximum sequence length.
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
    preset: Configuration preset:
        - "balanced": Intelligent selection (default)
        - "memory_efficient": Prioritize FSDP, CPU offload
        - "speed": Prioritize DDP, no offload
        - "conservative": Use safest options
    force_strategy: Force specific strategy (overrides auto-selection).
    verbose: Print configuration information.
    ddp_find_unused_parameters: DDP parameter for models with unused params.
    ddp_broadcast_buffers: DDP parameter for broadcasting buffers.
    ddp_bucket_cap_mb: DDP bucket size in MB.
    fsdp_min_num_params: FSDP minimum parameters for auto-wrapping.
    fsdp_transformer_layer_cls: FSDP transformer layer class for wrapping.
    fsdp_backward_prefetch: FSDP backward prefetch strategy.
    fsdp_limit_all_gathers: FSDP parameter for memory efficiency.
    fsdp_use_orig_params: FSDP parameter for using original parameters.

Returns:
    Configured trainer instance (Trainer, DDPTrainer, or FSDPTrainer).

Example:
    Basic usage with auto-detection:
    >>> trainer = AutoTrainer(
    ...     model=model,
    ...     optimizer=torch.optim.AdamW,
    ...     dataloader=dataloader,
    ...     device="auto",
    ...     context_length=2048,
    ...     learning_rate=3e-4
    ... )
    >>> trainer.train(epochs=10)

    Memory-efficient configuration:
    >>> trainer = AutoTrainer(
    ...     model=large_model,
    ...     optimizer=AdamW,
    ...     dataloader=dataloader,
    ...     device="auto",
    ...     preset="memory_efficient",  # Prioritize FSDP + CPU offload
    ...     context_length=2048
    ... )

    Custom device configuration:
    >>> from olm.train.device import DeviceConfig, TrainerStrategy
    >>> config = DeviceConfig(
    ...     num_gpus=4,
    ...     strategy=TrainerStrategy.MULTI_GPU_FSDP_FULL,
    ...     cuda_available=True
    ... )
    >>> trainer = AutoTrainer(model=model, device=config, ...)

    Force specific strategy:
    >>> trainer = AutoTrainer(
    ...     model=model,
    ...     device="auto",
    ...     force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    ...     ...
    ... )

### `auto_trainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str | olm.train.device.DeviceConfig = 'auto', context_length: int = 1024, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None, verbose: bool = True, ddp_find_unused_parameters: bool = False, ddp_broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, fsdp_min_num_params: int = 100000000, fsdp_transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, fsdp_backward_prefetch: str = 'BACKWARD_PRE', fsdp_limit_all_gathers: bool = True, fsdp_use_orig_params: bool = True) -> olm.train.trainer.trainer.Trainer | olm.train.trainer.ddp_trainer.DDPTrainer | olm.train.trainer.fsdp_trainer.FSDPTrainer`

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles all
the complexity of distributed training setup, device selection, and
parameter configuration.

Args:
    model: Model to train.
    optimizer: Optimizer instance or class.
    dataloader: DataLoader for training data.
    device: Device configuration. Options:
        - "auto": Full auto-detection (recommended)
        - "cuda:auto": Force CUDA with auto-configuration
        - "cpu:auto": Force CPU with auto-configuration
        - "cuda", "cuda:0", "cpu": Legacy device strings
        - DeviceConfig object: Custom configuration
    context_length: Maximum sequence length.
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
    preset: Configuration preset:
        - "balanced": Intelligent selection (default)
        - "memory_efficient": Prioritize FSDP, CPU offload
        - "speed": Prioritize DDP, no offload
        - "conservative": Use safest options
    force_strategy: Force specific strategy (overrides auto-selection).
    verbose: Print configuration information.
    ddp_find_unused_parameters: DDP parameter for models with unused params.
    ddp_broadcast_buffers: DDP parameter for broadcasting buffers.
    ddp_bucket_cap_mb: DDP bucket size in MB.
    fsdp_min_num_params: FSDP minimum parameters for auto-wrapping.
    fsdp_transformer_layer_cls: FSDP transformer layer class for wrapping.
    fsdp_backward_prefetch: FSDP backward prefetch strategy.
    fsdp_limit_all_gathers: FSDP parameter for memory efficiency.
    fsdp_use_orig_params: FSDP parameter for using original parameters.

Returns:
    Configured trainer instance (Trainer, DDPTrainer, or FSDPTrainer).

Example:
    Basic usage with auto-detection:
    >>> trainer = AutoTrainer(
    ...     model=model,
    ...     optimizer=torch.optim.AdamW,
    ...     dataloader=dataloader,
    ...     device="auto",
    ...     context_length=2048,
    ...     learning_rate=3e-4
    ... )
    >>> trainer.train(epochs=10)

    Memory-efficient configuration:
    >>> trainer = AutoTrainer(
    ...     model=large_model,
    ...     optimizer=AdamW,
    ...     dataloader=dataloader,
    ...     device="auto",
    ...     preset="memory_efficient",  # Prioritize FSDP + CPU offload
    ...     context_length=2048
    ... )

    Custom device configuration:
    >>> from olm.train.device import DeviceConfig, TrainerStrategy
    >>> config = DeviceConfig(
    ...     num_gpus=4,
    ...     strategy=TrainerStrategy.MULTI_GPU_FSDP_FULL,
    ...     cuda_available=True
    ... )
    >>> trainer = AutoTrainer(model=model, device=config, ...)

    Force specific strategy:
    >>> trainer = AutoTrainer(
    ...     model=model,
    ...     device="auto",
    ...     force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    ...     ...
    ... )

## Classes

### `CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)`

Callback to save model checkpoints at specified intervals.

Args:
    checkpoint_dir: Directory to save checkpoints.
    save_every: Save checkpoint every N steps.
    keep_last_n: Keep only the last N checkpoints.
    save_best: Whether to save the best model based on validation loss.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Save checkpoint after each optimization step if needed.

### `DDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, ddp_backend: str | None = None, find_unused_parameters: bool = False, broadcast_buffers: bool = True, bucket_cap_mb: int = 25, gradient_as_bucket_view: bool = True, static_graph: bool = False)`

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

- `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`
  Training loop with DDP support.

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

Callback to stop training early if validation loss doesn't improve.

Args:
    patience: Number of validation checks to wait for improvement.
    min_delta: Minimum change in validation loss to qualify as improvement.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Check for early stopping after each step.

### `FSDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, sharding_strategy: str = 'FULL_SHARD', auto_wrap_policy: str | None = 'size', min_num_params: int = 100000000, transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, cpu_offload: bool = False, backward_prefetch: str = 'BACKWARD_PRE', mixed_precision_policy: str | None = None, limit_all_gathers: bool = True, use_orig_params: bool = True)`

Trainer with PyTorch Fully Sharded Data Parallel (FSDP) support.

FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of larger models than DDP. Uses PyTorch's native FSDP.

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
    sharding_strategy: FSDP sharding strategy:
        - FULL_SHARD: Shard parameters, gradients, optimizer states (most memory efficient)
        - SHARD_GRAD_OP: Shard gradients and optimizer states only
        - NO_SHARD: Equivalent to DDP
        - HYBRID_SHARD: Full shard within node, replicate across nodes
    auto_wrap_policy: Policy for automatic module wrapping:
        - "size": Wrap based on parameter count (default, uses min_num_params)
        - "transformer": Wrap transformer layers (provide transformer_layer_cls)
        - None: Manual wrapping (model must already be wrapped)
    min_num_params: Minimum parameters for size-based wrapping (default: 1e8 = 100M).
    transformer_layer_cls: Transformer layer class for transformer wrapping policy.
    cpu_offload: Offload parameters to CPU when not in use.
    backward_prefetch: Prefetch parameters for backward pass (recommended).
    mixed_precision_policy: Mixed precision configuration (BF16, FP16, or None).
    limit_all_gathers: Limit all-gather operations for memory efficiency.
    use_orig_params: Use original parameters instead of flattened (better for optimizers).

Example:
    >>> # Launch with: torchrun --nproc_per_node=8 train.py
    >>> from olm.core.dist import setup_distributed
    >>> setup_distributed()
    >>>
    >>> trainer = FSDPTrainer(
    ...     model=model,
    ...     optimizer=torch.optim.AdamW,
    ...     dataloader=dataloader,
    ...     device=f"cuda:{get_local_rank()}",
    ...     context_length=2048,
    ...     learning_rate=3e-4,
    ...     sharding_strategy="FULL_SHARD",
    ...     auto_wrap_policy="size",
    ...     min_num_params=1e8,  # Wrap layers with 100M+ params
    ...     mixed_precision_policy="bf16"
    ... )
    >>> trainer.train(epochs=10)

#### Methods

- `save_checkpoint(self, path: str, state_dict_type: str = 'FULL_STATE_DICT') -> None`
  Save FSDP checkpoint.
- `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`
  Training loop with FSDP support.

### `LRMonitorCallback(log_every: int = 100)`

Callback to monitor and log learning rate.

Args:
    log_every: Log learning rate every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log learning rate after each optimization step if needed.

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

Callback to log metrics to a JSONL file.

Args:
    log_dir: Directory to save logs.
    log_every: Log metrics every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log metrics after each optimization step if needed.

### `ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)`

Callback to monitor training throughput (tokens/sec, samples/sec).

Args:
    log_every: Log throughput every N steps.
    context_length: Length of each sequence.
    batch_size: Total batch size (including gradient accumulation).

#### Methods

- `on_step_begin(self, trainer, step: int) -> None`
  Record start time of the step.
- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Calculate and log throughput.

### `Trainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True)`

Manages the training loop for Open Language Model (OLM) architectures.

This trainer handles the core training logic including:
- Automatic Mixed Precision (AMP) scaling
- Gradient accumulation
- Device management (moving data/models to GPU)
- Optimization steps
- Callbacks for validation, checkpointing, and custom logic
- Learning rate scheduling support
- Gradient clipping

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

- `add_callback(self, callback: olm.train.trainer.trainer.TrainerCallback) -> None`
  Add a callback to the trainer.
- `remove_callback(self, callback: olm.train.trainer.trainer.TrainerCallback) -> None`
  Remove a callback from the trainer.
- `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`
  Executes the training loop for a specified number of epochs.

### `TrainerCallback()`

Base class for trainer callbacks.

#### Methods

- `on_batch_begin(self, trainer: 'Trainer', batch_idx: int) -> None`
  Called at the beginning of each batch.
- `on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None`
  Called at the end of each batch.
- `on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None`
  Called at the beginning of each epoch.
- `on_epoch_end(self, trainer: 'Trainer', epoch: int) -> None`
  Called at the end of each epoch.
- `on_step_begin(self, trainer: 'Trainer', step: int) -> None`
  Called at the beginning of each optimization step (after gradient accumulation).
- `on_step_end(self, trainer: 'Trainer', step: int, loss: float) -> None`
  Called at the end of each optimization step.
- `on_train_begin(self, trainer: 'Trainer') -> None`
  Called at the beginning of training.
- `on_train_end(self, trainer: 'Trainer') -> None`
  Called at the end of training.

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

Callback to perform validation at specified intervals.

Args:
    val_dataloader: Validation dataloader.
    eval_every: Validate every N steps.
    device: Device to run validation on.
    use_amp: Whether to use automatic mixed precision.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Run validation after each optimization step if needed.
