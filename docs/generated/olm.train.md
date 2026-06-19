# `olm.train`

Training infrastructure for OLM.

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

### `detect_devices(verbose: bool = True) -> olm.train.device.DeviceConfig`

Detect available hardware and create device configuration.

Args:
    verbose: Print detection results

Returns:
    DeviceConfig with hardware information

Example:
    >>> config = detect_devices()
    >>> print(f"Found {config.num_gpus} GPUs")

### `determine_strategy(device_config: olm.train.device.DeviceConfig, model: torch.nn.modules.module.Module | None = None, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None) -> olm.train.device.DeviceConfig`

Determine optimal training strategy based on hardware and model.

Args:
    device_config: Device configuration from detect_devices()
    model: PyTorch model (optional, for memory estimation)
    preset: Configuration preset:
        - "balanced": Intelligent selection (default)
        - "memory_efficient": Prioritize FSDP, CPU offload
        - "speed": Prioritize DDP, no offload
        - "conservative": Use safest options
    force_strategy: Force specific strategy (overrides auto-selection)

Returns:
    Updated DeviceConfig with strategy and configuration

Example:
    >>> config = detect_devices()
    >>> config = determine_strategy(config, model=my_model)
    >>> print(f"Selected strategy: {config.strategy.value}")

### `estimate_model_size(model: torch.nn.modules.module.Module, verbose: bool = False) -> Dict[str, float]`

Estimate memory footprint of a model.

Args:
    model: PyTorch model
    verbose: Print estimation details

Returns:
    Dictionary with memory estimates in GB:
        - params_gb: Parameter memory
        - gradients_gb: Gradient memory
        - optimizer_gb: Optimizer state memory (assumes AdamW)
        - total_gb: Total estimated memory
        - num_params: Total number of parameters

Example:
    >>> memory = estimate_model_size(model)
    >>> print(f"Model requires ~{memory['total_gb']:.2f} GB")

### `parse_device_string(device: str, model: torch.nn.modules.module.Module | None = None) -> olm.train.device.DeviceConfig`

Parse device string and return configuration.

Supported formats:
    - "auto": Full auto-detection
    - "cuda:auto": Auto-detect CUDA configuration
    - "cpu:auto": Auto-detect CPU configuration
    - "cuda": Single CUDA device
    - "cuda:0": Specific CUDA device
    - "cpu": CPU device

Args:
    device: Device string
    model: Optional model for memory estimation

Returns:
    DeviceConfig

Example:
    >>> config = parse_device_string("auto", model=my_model)
    >>> config = parse_device_string("cuda:auto")

### `print_strategy_summary(config: olm.train.device.DeviceConfig) -> None`

Print a summary of the selected training strategy.

Args:
    config: Device configuration

## Classes

### `AdamW(params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False, fused: bool | None = None)`

AdamW optimizer with decoupled weight decay regularization.

This is a wrapper around PyTorch's built-in AdamW implementation from
"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
Unlike the original Adam, weight decay is applied directly to the parameters
rather than being added to the gradient.

This implementation is commonly used for training large language models and
transformers, offering better generalization than standard Adam.

Note: This class inherits from PyTorch's AdamW which ultimately inherits from
torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

Args:
    params: iterable of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-3)
    betas: coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999))
    eps: term added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay: weight decay coefficient (default: 0.01)
    amsgrad: whether to use the AMSGrad variant (default: False)
    maximize: maximize the params based on the objective, instead of minimizing (default: False)
    fused: whether to use the fused implementation (default: None, auto-detect)

Example:
    >>> model = nn.Linear(10, 5)
    >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    >>> optimizer.zero_grad()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> optimizer.step()

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

### `CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1)`

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

Args:
    optimizer: Wrapped optimizer.
    T_max: Maximum number of iterations (steps).
    eta_min: Minimum learning rate (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import CosineAnnealingLR
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    >>> for epoch in range(epochs):
    ...     train(...)
    ...     scheduler.step()

#### Methods

- `get_lr(self)`
  Compute learning rate using cosine annealing.

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

### `DeviceConfig(num_gpus: int, num_cpus: int, cuda_available: bool, gpu_memory_per_device: float | None = None, total_gpu_memory: float | None = None, strategy: olm.train.device.TrainerStrategy | None = None, device_type: str = 'cuda', local_rank: int = 0, world_size: int = 1, backend: str | None = None, mixed_precision: str | None = None, sharding_strategy: str | None = None, auto_wrap_policy: str | None = None, cpu_offload: bool = False) -> None`

Configuration for device and training strategy.

Attributes:
    num_gpus: Number of available GPUs
    num_cpus: Number of CPU cores
    cuda_available: Whether CUDA is available
    gpu_memory_per_device: GPU memory in GB per device
    total_gpu_memory: Total GPU memory in GB
    strategy: Selected training strategy
    device_type: Device type ('cuda' or 'cpu')
    local_rank: Local rank for distributed training
    world_size: World size for distributed training
    backend: Distributed backend ('nccl', 'gloo', or None)
    mixed_precision: Mixed precision dtype ('bf16', 'fp16', or None)
    sharding_strategy: FSDP sharding strategy (if applicable)
    auto_wrap_policy: FSDP auto wrap policy (if applicable)
    cpu_offload: Whether to offload parameters to CPU

#### Methods

- `to_dict(self) -> Dict[str, Any]`
  Convert config to dictionary.

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

### `LinearDecayLR(optimizer, total_steps: int, last_epoch: int = -1)`

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

Args:
    optimizer: Wrapped optimizer.
    total_steps: Total number of steps to decay over.
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import LinearDecayLR
    >>> scheduler = LinearDecayLR(optimizer, total_steps=1000)
    >>> for step in range(total_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

- `get_lr(self)`
  Compute learning rate using linear decay.

### `LinearLR(optimizer, total_steps: int, end_lr: float = 0, start_factor: float = 1.0, last_epoch: int = -1)`

Linear learning rate scheduler.

Linearly decreases (or increases) the learning rate from the initial
learning rate to end_lr over total_steps.

Args:
    optimizer: Wrapped optimizer.
    total_steps: Total number of steps for the schedule.
    end_lr: Target learning rate at the end (default: 0).
    start_factor: Initial learning rate multiplier (default: 1.0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import LinearLR
    >>> # Decay from initial LR to 0
    >>> scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
    >>> for step in range(total_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

- `get_lr(self)`
  Compute learning rate using linear interpolation.

### `Lion(params: Iterable, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False)`

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from "Symbolic Discovery of Optimization Algorithms"
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

Args:
    params: iterable of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
    betas: coefficients used for computing running averages (default: (0.9, 0.99))
    weight_decay: weight decay coefficient (default: 0.0)
    use_triton: whether to use Triton kernel for faster computation (default: False)

Example:
    >>> model = nn.Linear(10, 5)
    >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
    >>> optimizer.zero_grad()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> optimizer.step()

#### Methods

- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized tensors to zero.

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

Callback to log metrics to a JSONL file.

Args:
    log_dir: Directory to save logs.
    log_every: Log metrics every N steps.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Log metrics after each optimization step if needed.

### `OptimizerBase(params: collections.abc.Iterable[torch.Tensor] | collections.abc.Iterable[dict[str, Any]] | collections.abc.Iterable[tuple[str, torch.Tensor]], defaults: dict[str, typing.Any]) -> None`

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch's Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### Methods

- `extra_repr(self) -> str`
  String representation of the optimizer for debugging.
- `load_state_dict(self, state_dict: Dict[str, Any])`
  Loads the optimizer state.
- `state_dict(self) -> Dict[str, Any]`
  Returns the state of the optimizer as a dict.
- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized tensors to zero or None.

### `SchedulerBase(optimizer, last_epoch: int = -1, verbose: bool = False)`

Base class for all OLM learning rate schedulers.

This class extends PyTorch's _LRScheduler and provides a consistent
interface for implementing custom learning rate schedules. All OLM
schedulers should inherit from this class to maintain uniformity.

Subclasses must implement:
    - get_lr(): Compute the learning rate for the current step
    - _get_closed_form_lr() (optional): Closed-form solution for efficiency

Args:
    optimizer: Wrapped PyTorch optimizer.
    last_epoch: The index of the last epoch (default: -1).
    verbose: If True, prints a message to stdout for each update (default: False).

Example:
    >>> class MyScheduler(SchedulerBase):
    ...     def __init__(self, optimizer, param, last_epoch=-1):
    ...         self.param = param
    ...         super().__init__(optimizer, last_epoch)
    ...
    ...     def get_lr(self):
    ...         # Custom logic here
    ...         return [base_lr * self.param for base_lr in self.base_lrs]

#### Methods

- `get_last_lr(self) -> List[float]`
  Return last computed learning rate by current scheduler.
- `get_lr(self) -> List[float]`
  Compute learning rate for each parameter group.
- `load_state_dict(self, state_dict)`
  Load the scheduler state from a checkpoint.
- `state_dict(self)`
  Returns the state of the scheduler as a dict.

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

### `TrainerStrategy(*values)`

Training strategy based on available hardware.

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

### `WarmupCosineScheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0, last_epoch: int = -1)`

Combined warmup and cosine annealing scheduler.

Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
then applies cosine annealing decay to min_lr over the remaining steps.

Args:
    optimizer: Wrapped optimizer.
    warmup_steps: Number of warmup steps.
    total_steps: Total number of training steps.
    min_lr: Minimum learning rate after decay (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
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

#### Methods

- `get_lr(self)`
  Compute learning rate with warmup and cosine decay.

### `WarmupLR(optimizer, warmup_steps: int, start_lr: float = 0, last_epoch: int = -1)`

Learning rate warmup scheduler.

Linearly increases the learning rate from 0 to the base learning rate
over warmup_steps.

Args:
    optimizer: Wrapped optimizer.
    warmup_steps: Number of warmup steps.
    start_lr: Initial learning rate (default: 0).
    last_epoch: The index of last epoch (default: -1).

Example:
    >>> from olm.train.schedulers import WarmupLR
    >>> scheduler = WarmupLR(optimizer, warmup_steps=1000)
    >>> for step in range(warmup_steps):
    ...     train(...)
    ...     scheduler.step()

#### Methods

- `get_lr(self)`
  Compute learning rate during warmup.
