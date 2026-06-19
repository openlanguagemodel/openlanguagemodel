"""
Automatic trainer selection based on hardware configuration.

This module provides the AutoTrainer factory that automatically selects
and configures the appropriate trainer (Trainer, DDPTrainer, FSDPTrainer)
based on available hardware and model characteristics.
"""

from typing import Type, Optional, List, Union, Any
import torch
import torch.nn as nn
from torch.distributed.fsdp import ShardingStrategy

from olm.train.trainer.trainer import Trainer, TrainerCallback
from olm.train.trainer.ddp_trainer import DDPTrainer
from olm.train.trainer.fsdp_trainer import FSDPTrainer
from olm.train.losses.base import LossBase
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.data.datasets import DataLoader
from olm.train.device import (
    DeviceConfig,
    TrainerStrategy,
    detect_devices,
    determine_strategy,
    parse_device_string,
    print_strategy_summary,
    estimate_model_size,
)
from olm.core.dist import setup_distributed, is_distributed, get_local_rank


def AutoTrainer(
    model: nn.Module,
    optimizer: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer]],
    dataloader: DataLoader,
    device: Union[str, DeviceConfig] = "auto",
    context_length: int = 1024,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    loss: Type[LossBase] = CrossEntropyLoss,
    callbacks: Optional[List[TrainerCallback]] = None,
    scheduler: Optional[Any] = None,
    grad_clip_norm: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    min_lr: float = 0.0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    use_warmup_cosine: bool = True,
    # Auto-configuration parameters
    preset: str = "balanced",
    force_strategy: Optional[TrainerStrategy] = None,
    verbose: bool = True,
    # Advanced DDP parameters (used if DDP is selected)
    ddp_find_unused_parameters: bool = False,
    ddp_broadcast_buffers: bool = True,
    ddp_bucket_cap_mb: int = 25,
    # Advanced FSDP parameters (used if FSDP is selected)
    fsdp_min_num_params: int = int(1e8),
    fsdp_transformer_layer_cls: Optional[Type[nn.Module]] = None,
    fsdp_backward_prefetch: str = "BACKWARD_PRE",
    fsdp_limit_all_gathers: bool = True,
    fsdp_use_orig_params: bool = True,
) -> Union[Trainer, DDPTrainer, FSDPTrainer]:
    """
    Automatically select and configure the optimal trainer based on hardware.

    This factory function intelligently chooses between Trainer, DDPTrainer,
    and FSDPTrainer based on available GPUs and model size. It handles all
    the complexity of distributed training setup, device selection, and
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
    """
    # Parse device configuration
    requested_device = device.lower() if isinstance(device, str) else None
    if isinstance(device, str):
        device_config = parse_device_string(device, model=model)
        if requested_device in {"auto", "cuda:auto"}:
            # Auto mode: determine strategy
            device_config = determine_strategy(
                device_config,
                model=model,
                preset=preset,
                force_strategy=force_strategy,
            )
    elif isinstance(device, DeviceConfig):
        device_config = device
        # If strategy not set, determine it
        if device_config.strategy is None:
            device_config = determine_strategy(
                device_config,
                model=model,
                preset=preset,
                force_strategy=force_strategy,
            )
    else:
        raise ValueError(
            f"Invalid device type: {type(device)}. " f"Expected str or DeviceConfig."
        )

    # Print configuration if verbose
    if verbose and device_config.strategy:
        if device != "auto" and not isinstance(device, DeviceConfig):
            # Only print for auto modes
            pass
        else:
            print_strategy_summary(device_config)
            if model is not None:
                estimate_model_size(model, verbose=True)

    # Setup distributed training if needed
    strategy = device_config.strategy
    if strategy in [
        TrainerStrategy.MULTI_GPU_DDP,
        TrainerStrategy.MULTI_GPU_FSDP_HYBRID,
        TrainerStrategy.MULTI_GPU_FSDP_FULL,
    ]:
        if not is_distributed():
            setup_distributed(backend=device_config.backend)

    # Determine device string for trainer
    if device_config.device_type == "cuda":
        if is_distributed():
            trainer_device = f"cuda:{get_local_rank()}"
        elif (
            requested_device is not None
            and requested_device.startswith("cuda")
            and requested_device != "cuda:auto"
        ):
            trainer_device = requested_device
        else:
            trainer_device = "cuda"
    else:
        trainer_device = "cpu"

    # Select and instantiate appropriate trainer
    if strategy == TrainerStrategy.SINGLE_CPU or strategy == TrainerStrategy.SINGLE_GPU:
        # Single device training - use base Trainer
        if verbose:
            print(f"Initializing single-device Trainer (device: {trainer_device})")

        return Trainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=trainer_device,
            context_length=context_length,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp and device_config.device_type == "cuda",
            loss=loss,
            callbacks=callbacks,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_warmup_cosine=use_warmup_cosine,
        )

    elif strategy == TrainerStrategy.MULTI_GPU_DDP:
        # Distributed Data Parallel
        if verbose:
            print(f"Initializing DDPTrainer (backend: {device_config.backend})")

        return DDPTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=trainer_device,
            context_length=context_length,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            loss=loss,
            callbacks=callbacks,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_warmup_cosine=use_warmup_cosine,
            ddp_backend=device_config.backend,
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=ddp_broadcast_buffers,
            bucket_cap_mb=ddp_bucket_cap_mb,
        )

    elif strategy in [
        TrainerStrategy.MULTI_GPU_FSDP_HYBRID,
        TrainerStrategy.MULTI_GPU_FSDP_FULL,
    ]:
        # Fully Sharded Data Parallel
        if verbose:
            print(
                f"Initializing FSDPTrainer "
                f"(sharding: {device_config.sharding_strategy})"
            )

        # Configure CPU offload based on device config
        cpu_offload = device_config.cpu_offload

        # Configure mixed precision
        mixed_precision = device_config.mixed_precision

        # Configure auto wrap policy
        auto_wrap_policy = device_config.auto_wrap_policy or "size"

        return FSDPTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=trainer_device,
            context_length=context_length,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            loss=loss,
            callbacks=callbacks,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_warmup_cosine=use_warmup_cosine,
            sharding_strategy=device_config.sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            min_num_params=fsdp_min_num_params,
            transformer_layer_cls=fsdp_transformer_layer_cls,
            cpu_offload=cpu_offload,
            backward_prefetch=fsdp_backward_prefetch,
            mixed_precision_policy=mixed_precision,
            limit_all_gathers=fsdp_limit_all_gathers,
            use_orig_params=fsdp_use_orig_params,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Convenience alias
auto_trainer = AutoTrainer
