# `olm.train.device`

Device detection and automatic trainer strategy selection.

This module provides automatic hardware detection and intelligent selection
of the optimal training strategy (single GPU, DDP, FSDP) based on available
resources and model characteristics.

## Functions

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

### `TrainerStrategy(*values)`

Training strategy based on available hardware.
