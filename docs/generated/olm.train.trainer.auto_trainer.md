# `olm.train.trainer.auto_trainer`

Source: [`src/olm/train/trainer/auto_trainer.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/auto_trainer.py#L1)

Automatic trainer selection based on hardware configuration.

This module provides the AutoTrainer factory that automatically selects
and configures the appropriate trainer (Trainer, DDPTrainer, FSDPTrainer)
based on available hardware and model characteristics.

## Functions

### `AutoTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str | olm.train.device.DeviceConfig = 'auto', context_length: int = 1024, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None, verbose: bool = True, ddp_find_unused_parameters: bool = False, ddp_broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, fsdp_min_num_params: int = 100000000, fsdp_transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, fsdp_backward_prefetch: str = 'BACKWARD_PRE', fsdp_limit_all_gathers: bool = True, fsdp_use_orig_params: bool = True) -> olm.train.trainer.trainer.Trainer | olm.train.trainer.ddp_trainer.DDPTrainer | olm.train.trainer.fsdp_trainer.FSDPTrainer`

Source: [`src/olm/train/trainer/auto_trainer.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/auto_trainer.py#L32)

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles the
single-node multi-GPU setup, device selection, and
parameter configuration.

Forward/training contract:
    The model is expected to accept ``input_ids`` shaped
    ``[batch, context_length]`` and return logits shaped
    ``[batch, context_length, vocab_size]``. The dataloader should yield
    ``(input_ids, labels)`` where both tensors are shaped
    ``[batch, context_length]``.

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

Source: [`src/olm/train/trainer/auto_trainer.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/auto_trainer.py#L32)

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles the
single-node multi-GPU setup, device selection, and
parameter configuration.

Forward/training contract:
    The model is expected to accept ``input_ids`` shaped
    ``[batch, context_length]`` and return logits shaped
    ``[batch, context_length, vocab_size]``. The dataloader should yield
    ``(input_ids, labels)`` where both tensors are shaped
    ``[batch, context_length]``.

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
