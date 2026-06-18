"""
Device detection and automatic trainer strategy selection.

This module provides automatic hardware detection and intelligent selection
of the optimal training strategy (single GPU, DDP, FSDP) based on available
resources and model characteristics.
"""

import os
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import warnings


class TrainerStrategy(Enum):
    """Training strategy based on available hardware."""

    SINGLE_CPU = "single_cpu"
    SINGLE_GPU = "single_gpu"
    MULTI_GPU_DDP = "multi_gpu_ddp"
    MULTI_GPU_FSDP_HYBRID = "multi_gpu_fsdp_hybrid"
    MULTI_GPU_FSDP_FULL = "multi_gpu_fsdp_full"


@dataclass
class DeviceConfig:
    """
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
    """

    num_gpus: int
    num_cpus: int
    cuda_available: bool
    gpu_memory_per_device: Optional[float] = None
    total_gpu_memory: Optional[float] = None
    strategy: Optional[TrainerStrategy] = None
    device_type: str = "cuda"
    local_rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None
    mixed_precision: Optional[str] = None
    sharding_strategy: Optional[str] = None
    auto_wrap_policy: Optional[str] = None
    cpu_offload: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "cuda_available": self.cuda_available,
            "gpu_memory_per_device": self.gpu_memory_per_device,
            "total_gpu_memory": self.total_gpu_memory,
            "strategy": self.strategy.value if self.strategy else None,
            "device_type": self.device_type,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "mixed_precision": self.mixed_precision,
            "sharding_strategy": self.sharding_strategy,
            "auto_wrap_policy": self.auto_wrap_policy,
            "cpu_offload": self.cpu_offload,
        }


def detect_devices(verbose: bool = True) -> DeviceConfig:
    """
    Detect available hardware and create device configuration.

    Args:
        verbose: Print detection results

    Returns:
        DeviceConfig with hardware information

    Example:
        >>> config = detect_devices()
        >>> print(f"Found {config.num_gpus} GPUs")
    """
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0
    num_cpus = os.cpu_count() or 1

    gpu_memory_per_device = None
    total_gpu_memory = None

    if cuda_available and num_gpus > 0:
        # Get GPU memory in GB
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_per_device = gpu_memory_bytes / (1024**3)  # Convert to GB
        total_gpu_memory = gpu_memory_per_device * num_gpus

    # Check if running in distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    config = DeviceConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cuda_available=cuda_available,
        gpu_memory_per_device=gpu_memory_per_device,
        total_gpu_memory=total_gpu_memory,
        device_type="cuda" if cuda_available else "cpu",
        local_rank=local_rank,
        world_size=world_size,
    )

    if verbose:
        print("=" * 70)
        print("Device Detection Results:")
        print("-" * 70)
        print(f"  CUDA Available: {cuda_available}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Number of CPUs: {num_cpus}")
        if cuda_available and num_gpus > 0:
            print(f"  GPU Memory per Device: {gpu_memory_per_device:.2f} GB")
            print(f"  Total GPU Memory: {total_gpu_memory:.2f} GB")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        if world_size > 1:
            print(f"  Distributed Training: Yes (World Size: {world_size})")
            print(f"  Local Rank: {local_rank}")
        else:
            print(f"  Distributed Training: No")
        print("=" * 70)

    return config


def estimate_model_size(
    model: torch.nn.Module, verbose: bool = False
) -> Dict[str, float]:
    """
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
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (assuming float32, 4 bytes per param)
    bytes_per_param = 4
    params_gb = (num_params * bytes_per_param) / (1024**3)

    # Gradients same size as parameters
    gradients_gb = params_gb

    # Optimizer states (AdamW has 2 states per param: momentum and variance)
    optimizer_gb = params_gb * 2

    # Total memory during training
    total_gb = params_gb + gradients_gb + optimizer_gb

    # Add activation memory estimate (rough: 20% of model size per batch)
    activation_gb = params_gb * 0.2
    total_with_activations = total_gb + activation_gb

    result = {
        "num_params": num_params,
        "num_trainable": num_trainable,
        "params_gb": params_gb,
        "gradients_gb": gradients_gb,
        "optimizer_gb": optimizer_gb,
        "activation_gb": activation_gb,
        "total_gb": total_gb,
        "total_with_activations": total_with_activations,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("Model Memory Estimation:")
        print("-" * 70)
        print(f"  Total Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Trainable Parameters: {num_trainable:,} ({num_trainable/1e6:.2f}M)")
        print(f"  Parameter Memory: {params_gb:.2f} GB")
        print(f"  Gradient Memory: {gradients_gb:.2f} GB")
        print(f"  Optimizer Memory: {optimizer_gb:.2f} GB")
        print(f"  Estimated Activation Memory: {activation_gb:.2f} GB")
        print(f"  Total Training Memory: {total_with_activations:.2f} GB")
        print("=" * 70 + "\n")

    return result


def determine_strategy(
    device_config: DeviceConfig,
    model: Optional[torch.nn.Module] = None,
    preset: str = "balanced",
    force_strategy: Optional[TrainerStrategy] = None,
) -> DeviceConfig:
    """
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
    """
    if force_strategy is not None:
        device_config.strategy = force_strategy
        _configure_strategy_params(device_config, preset)
        return device_config

    num_gpus = device_config.num_gpus
    gpu_memory = device_config.gpu_memory_per_device or 0

    # Estimate model size if provided
    model_memory_gb = 0
    if model is not None:
        memory_info = estimate_model_size(model, verbose=False)
        model_memory_gb = memory_info["total_with_activations"]

    # Decision logic
    if num_gpus == 0:
        device_config.strategy = TrainerStrategy.SINGLE_CPU
        device_config.device_type = "cpu"
        device_config.backend = "gloo"

    elif num_gpus == 1:
        # Single GPU
        if model is not None and model_memory_gb > gpu_memory * 0.8:
            warnings.warn(
                f"Model memory ({model_memory_gb:.2f} GB) may exceed GPU memory "
                f"({gpu_memory:.2f} GB). Consider using gradient checkpointing or "
                f"a smaller model.",
                RuntimeWarning,
            )
        device_config.strategy = TrainerStrategy.SINGLE_GPU
        device_config.device_type = "cuda"
        device_config.backend = None  # No distributed backend needed

    else:
        # Multi-GPU: Choose between DDP, FSDP_HYBRID, FSDP_FULL
        device_config.backend = "nccl"  # Use NCCL for multi-GPU

        if preset == "memory_efficient":
            # Always use FSDP for maximum memory efficiency
            if num_gpus <= 4:
                device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_HYBRID
            else:
                device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_FULL

        elif preset == "speed":
            # Prefer DDP for speed unless model is too large
            if model is None or model_memory_gb < gpu_memory * 0.6:
                device_config.strategy = TrainerStrategy.MULTI_GPU_DDP
            else:
                device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_HYBRID

        else:  # balanced or conservative
            # Intelligent selection based on model size and GPU count
            if model is None:
                # No model provided, use conservative defaults
                if num_gpus <= 2:
                    device_config.strategy = TrainerStrategy.MULTI_GPU_DDP
                else:
                    device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_HYBRID
            else:
                # Model-aware selection
                memory_per_gpu_with_ddp = model_memory_gb / num_gpus

                if memory_per_gpu_with_ddp < gpu_memory * 0.6:
                    # Model fits comfortably with DDP
                    device_config.strategy = TrainerStrategy.MULTI_GPU_DDP
                elif num_gpus <= 4:
                    # Use hybrid sharding for small clusters
                    device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_HYBRID
                else:
                    # Use full sharding for large clusters or big models
                    device_config.strategy = TrainerStrategy.MULTI_GPU_FSDP_FULL

    # Configure strategy-specific parameters
    _configure_strategy_params(device_config, preset)

    return device_config


def _configure_strategy_params(config: DeviceConfig, preset: str) -> None:
    """Configure strategy-specific parameters based on preset."""

    # Mixed precision configuration
    if config.cuda_available:
        # Check if BF16 is available (Ampere GPUs and newer)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            config.mixed_precision = "bf16"
        else:
            config.mixed_precision = "fp16"
    else:
        config.mixed_precision = None

    # FSDP-specific configuration
    if "fsdp" in config.strategy.value:
        if config.strategy == TrainerStrategy.MULTI_GPU_FSDP_HYBRID:
            config.sharding_strategy = "HYBRID_SHARD"
        else:
            config.sharding_strategy = "FULL_SHARD"

        # Auto wrap policy
        config.auto_wrap_policy = "size"  # Default to size-based

        # CPU offload based on preset
        if preset == "memory_efficient":
            config.cpu_offload = True
        else:
            config.cpu_offload = False


def parse_device_string(
    device: str, model: Optional[torch.nn.Module] = None
) -> DeviceConfig:
    """
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
    """
    device_lower = device.lower()

    if device_lower == "auto":
        # Full auto-detection
        config = detect_devices(verbose=True)
        config = determine_strategy(config, model=model, preset="balanced")
        return config

    elif device_lower == "cuda:auto":
        # Force CUDA, but auto-configure
        config = detect_devices(verbose=True)
        if not config.cuda_available:
            raise RuntimeError("CUDA not available, cannot use 'cuda:auto'")
        config.device_type = "cuda"
        config = determine_strategy(config, model=model, preset="balanced")
        return config

    elif device_lower == "cpu:auto":
        # Force CPU, but auto-configure
        config = detect_devices(verbose=True)
        config.device_type = "cpu"
        config.strategy = TrainerStrategy.SINGLE_CPU
        config.backend = "gloo" if config.world_size > 1 else None
        return config

    else:
        # Legacy device string (cuda, cuda:0, cpu, etc.)
        # Return minimal config for backward compatibility
        config = detect_devices(verbose=False)
        if device_lower.startswith("cuda"):
            config.device_type = "cuda"
            config.strategy = TrainerStrategy.SINGLE_GPU
        else:
            config.device_type = "cpu"
            config.strategy = TrainerStrategy.SINGLE_CPU
        return config


def print_strategy_summary(config: DeviceConfig) -> None:
    """
    Print a summary of the selected training strategy.

    Args:
        config: Device configuration
    """
    print("\n" + "=" * 70)
    print("Training Strategy Selection:")
    print("-" * 70)
    print(f"  Strategy: {config.strategy.value if config.strategy else 'None'}")
    print(f"  Device Type: {config.device_type}")
    print(f"  Number of GPUs: {config.num_gpus}")

    if config.backend:
        print(f"  Backend: {config.backend}")

    if config.mixed_precision:
        print(f"  Mixed Precision: {config.mixed_precision}")

    if config.sharding_strategy:
        print(f"  Sharding Strategy: {config.sharding_strategy}")

    if config.auto_wrap_policy:
        print(f"  Auto Wrap Policy: {config.auto_wrap_policy}")

    if config.cpu_offload:
        print(f"  CPU Offload: Enabled")

    if config.world_size > 1:
        print(f"  World Size: {config.world_size}")
        print(f"  Local Rank: {config.local_rank}")

    print("=" * 70 + "\n")
