# `olm.train`

Source: [`src/olm/train/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/__init__.py#L1)

Training infrastructure for OLM.

## Functions

### `AutoTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str | olm.train.device.DeviceConfig = 'auto', context_length: int = 1024, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None, verbose: bool = True, ddp_find_unused_parameters: bool = False, ddp_broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, fsdp_min_num_params: int = 100000000, fsdp_transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, fsdp_backward_prefetch: str = 'BACKWARD_PRE', fsdp_limit_all_gathers: bool = True, fsdp_use_orig_params: bool = True) -> olm.train.trainer.trainer.Trainer | olm.train.trainer.ddp_trainer.DDPTrainer | olm.train.trainer.fsdp_trainer.FSDPTrainer`

Source: [`src/olm/train/trainer/auto_trainer.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/auto_trainer.py#L32)

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles the
single-node multi-GPU setup, device selection, and
parameter configuration.

**Forward / Training Contract**

The model is expected to accept ``input_ids`` shaped
``[batch, context_length]`` and return logits shaped
``[batch, context_length, vocab_size]``. The dataloader should yield
``(input_ids, labels)`` where both tensors are shaped
``[batch, context_length]``.

**Parameters**

- `model`: Model to train.
- `optimizer`: Optimizer instance or class.
- `dataloader`: DataLoader for training data.
- `device`: Device configuration. Options: - "auto": Full auto-detection (recommended) - "cuda:auto": Force CUDA with auto-configuration - "cpu:auto": Force CPU with auto-configuration - "cuda", "cuda:0", "cpu": Legacy device strings - DeviceConfig object: Custom configuration
- `context_length`: Maximum sequence length.
- `grad_accum_steps`: Gradient accumulation steps.
- `use_amp`: Use automatic mixed precision.
- `loss`: Loss function class.
- `callbacks`: Training callbacks.
- `scheduler`: Learning rate scheduler.
- `grad_clip_norm`: Gradient clipping threshold.
- `warmup_steps`: Warmup steps for scheduler.
- `total_steps`: Total training steps.
- `min_lr`: Minimum learning rate.
- `learning_rate`: Learning rate (if optimizer is a class).
- `weight_decay`: Weight decay (if optimizer is a class).
- `use_warmup_cosine`: Use warmup+cosine scheduler by default.
- `preset`: Configuration preset: - "balanced": Intelligent selection (default) - "memory_efficient": Prioritize FSDP, CPU offload - "speed": Prioritize DDP, no offload - "conservative": Use safest options
- `force_strategy`: Force specific strategy (overrides auto-selection).
- `verbose`: Print configuration information.
- `ddp_find_unused_parameters`: DDP parameter for models with unused params.
- `ddp_broadcast_buffers`: DDP parameter for broadcasting buffers.
- `ddp_bucket_cap_mb`: DDP bucket size in MB.
- `fsdp_min_num_params`: FSDP minimum parameters for auto-wrapping.
- `fsdp_transformer_layer_cls`: FSDP transformer layer class for wrapping.
- `fsdp_backward_prefetch`: FSDP backward prefetch strategy.
- `fsdp_limit_all_gathers`: FSDP parameter for memory efficiency.
- `fsdp_use_orig_params`: FSDP parameter for using original parameters.

**Returns**

Configured trainer instance (Trainer, DDPTrainer, or FSDPTrainer).

**Example**

```python
# Basic usage with auto-detection:
trainer = AutoTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=dataloader,
    device="auto",
    context_length=2048,
    learning_rate=3e-4
)
trainer.train(epochs=10)

# Memory-efficient configuration:
trainer = AutoTrainer(
    model=large_model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    preset="memory_efficient",  # Prioritize FSDP + CPU offload
    context_length=2048
)

# Custom device configuration:
from olm.train.device import DeviceConfig, TrainerStrategy
config = DeviceConfig(
    num_gpus=4,
    strategy=TrainerStrategy.MULTI_GPU_FSDP_FULL,
    cuda_available=True
)
trainer = AutoTrainer(model=model, device=config, ...)

# Force specific strategy:
trainer = AutoTrainer(
    model=model,
    device="auto",
    force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    ...
)
```

### `auto_trainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str | olm.train.device.DeviceConfig = 'auto', context_length: int = 1024, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None, verbose: bool = True, ddp_find_unused_parameters: bool = False, ddp_broadcast_buffers: bool = True, ddp_bucket_cap_mb: int = 25, fsdp_min_num_params: int = 100000000, fsdp_transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, fsdp_backward_prefetch: str = 'BACKWARD_PRE', fsdp_limit_all_gathers: bool = True, fsdp_use_orig_params: bool = True) -> olm.train.trainer.trainer.Trainer | olm.train.trainer.ddp_trainer.DDPTrainer | olm.train.trainer.fsdp_trainer.FSDPTrainer`

Source: [`src/olm/train/trainer/auto_trainer.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/auto_trainer.py#L32)

Automatically select and configure the optimal trainer based on hardware.

This factory function intelligently chooses between Trainer, DDPTrainer,
and FSDPTrainer based on available GPUs and model size. It handles the
single-node multi-GPU setup, device selection, and
parameter configuration.

**Forward / Training Contract**

The model is expected to accept ``input_ids`` shaped
``[batch, context_length]`` and return logits shaped
``[batch, context_length, vocab_size]``. The dataloader should yield
``(input_ids, labels)`` where both tensors are shaped
``[batch, context_length]``.

**Parameters**

- `model`: Model to train.
- `optimizer`: Optimizer instance or class.
- `dataloader`: DataLoader for training data.
- `device`: Device configuration. Options: - "auto": Full auto-detection (recommended) - "cuda:auto": Force CUDA with auto-configuration - "cpu:auto": Force CPU with auto-configuration - "cuda", "cuda:0", "cpu": Legacy device strings - DeviceConfig object: Custom configuration
- `context_length`: Maximum sequence length.
- `grad_accum_steps`: Gradient accumulation steps.
- `use_amp`: Use automatic mixed precision.
- `loss`: Loss function class.
- `callbacks`: Training callbacks.
- `scheduler`: Learning rate scheduler.
- `grad_clip_norm`: Gradient clipping threshold.
- `warmup_steps`: Warmup steps for scheduler.
- `total_steps`: Total training steps.
- `min_lr`: Minimum learning rate.
- `learning_rate`: Learning rate (if optimizer is a class).
- `weight_decay`: Weight decay (if optimizer is a class).
- `use_warmup_cosine`: Use warmup+cosine scheduler by default.
- `preset`: Configuration preset: - "balanced": Intelligent selection (default) - "memory_efficient": Prioritize FSDP, CPU offload - "speed": Prioritize DDP, no offload - "conservative": Use safest options
- `force_strategy`: Force specific strategy (overrides auto-selection).
- `verbose`: Print configuration information.
- `ddp_find_unused_parameters`: DDP parameter for models with unused params.
- `ddp_broadcast_buffers`: DDP parameter for broadcasting buffers.
- `ddp_bucket_cap_mb`: DDP bucket size in MB.
- `fsdp_min_num_params`: FSDP minimum parameters for auto-wrapping.
- `fsdp_transformer_layer_cls`: FSDP transformer layer class for wrapping.
- `fsdp_backward_prefetch`: FSDP backward prefetch strategy.
- `fsdp_limit_all_gathers`: FSDP parameter for memory efficiency.
- `fsdp_use_orig_params`: FSDP parameter for using original parameters.

**Returns**

Configured trainer instance (Trainer, DDPTrainer, or FSDPTrainer).

**Example**

```python
# Basic usage with auto-detection:
trainer = AutoTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=dataloader,
    device="auto",
    context_length=2048,
    learning_rate=3e-4
)
trainer.train(epochs=10)

# Memory-efficient configuration:
trainer = AutoTrainer(
    model=large_model,
    optimizer=AdamW,
    dataloader=dataloader,
    device="auto",
    preset="memory_efficient",  # Prioritize FSDP + CPU offload
    context_length=2048
)

# Custom device configuration:
from olm.train.device import DeviceConfig, TrainerStrategy
config = DeviceConfig(
    num_gpus=4,
    strategy=TrainerStrategy.MULTI_GPU_FSDP_FULL,
    cuda_available=True
)
trainer = AutoTrainer(model=model, device=config, ...)

# Force specific strategy:
trainer = AutoTrainer(
    model=model,
    device="auto",
    force_strategy=TrainerStrategy.MULTI_GPU_DDP,
    ...
)
```

### `detect_devices(verbose: bool = True) -> olm.train.device.DeviceConfig`

Source: [`src/olm/train/device.py:84`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L84)

Detect available hardware and create device configuration.

**Parameters**

- `verbose`: Print detection results

**Returns**

DeviceConfig with hardware information

**Example**

```python
config = detect_devices()
print(f"Found {config.num_gpus} GPUs")
```

### `determine_strategy(device_config: olm.train.device.DeviceConfig, model: torch.nn.modules.module.Module | None = None, preset: str = 'balanced', force_strategy: olm.train.device.TrainerStrategy | None = None) -> olm.train.device.DeviceConfig`

Source: [`src/olm/train/device.py:217`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L217)

Determine optimal training strategy based on hardware and model.

**Parameters**

- `device_config`: Device configuration from detect_devices()
- `model`: PyTorch model (optional, for memory estimation)
- `preset`: Configuration preset: - "balanced": Intelligent selection (default) - "memory_efficient": Prioritize FSDP, CPU offload - "speed": Prioritize DDP, no offload - "conservative": Use safest options
- `force_strategy`: Force specific strategy (overrides auto-selection)

**Returns**

Updated DeviceConfig with strategy and configuration

**Example**

```python
config = detect_devices()
config = determine_strategy(config, model=my_model)
print(f"Selected strategy: {config.strategy.value}")
```

### `estimate_model_size(model: torch.nn.modules.module.Module, verbose: bool = False) -> Dict[str, float]`

Source: [`src/olm/train/device.py:147`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L147)

Estimate memory footprint of a model.

**Parameters**

- `model`: PyTorch model
- `verbose`: Print estimation details

**Returns**

Dictionary with memory estimates in GB:
- params_gb: Parameter memory
- gradients_gb: Gradient memory
- optimizer_gb: Optimizer state memory (assumes AdamW)
- total_gb: Total estimated memory
- num_params: Total number of parameters

**Example**

```python
memory = estimate_model_size(model)
print(f"Model requires ~{memory['total_gb']:.2f} GB")
```

### `parse_device_string(device: str, model: torch.nn.modules.module.Module | None = None) -> olm.train.device.DeviceConfig`

Source: [`src/olm/train/device.py:353`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L353)

Parse device string and return configuration.

Supported formats:
    - "auto": Full auto-detection
    - "cuda:auto": Auto-detect CUDA configuration
    - "cpu:auto": Auto-detect CPU configuration
    - "cuda": Single CUDA device
    - "cuda:0": Specific CUDA device
    - "cpu": CPU device

**Parameters**

- `device`: Device string
- `model`: Optional model for memory estimation

**Returns**

DeviceConfig

**Example**

```python
config = parse_device_string("auto", model=my_model)
config = parse_device_string("cuda:auto")
```

### `print_strategy_summary(config: olm.train.device.DeviceConfig) -> None`

Source: [`src/olm/train/device.py:416`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L416)

Print a summary of the selected training strategy.

**Parameters**

- `config`: Device configuration

## Classes

### `AdamW(params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False, fused: bool | None = None)`

**Bases:** `AdamW`

Source: [`src/olm/train/optim/adamw.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/adamw.py#L7)

AdamW optimizer with decoupled weight decay regularization.

This is a wrapper around PyTorch's built-in AdamW implementation from
"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
Unlike the original Adam, weight decay is applied directly to the parameters
rather than being added to the gradient.

This implementation is commonly used for training large language models and
transformers, offering better generalization than standard Adam.

Note: This class inherits from PyTorch's AdamW which ultimately inherits from
torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

**Parameters**

- `params`: iterable of parameters to optimize or dicts defining parameter groups
- `lr`: learning rate (default: 1e-3)
- `betas`: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
- `eps`: term added to the denominator to improve numerical stability (default: 1e-8)
- `weight_decay`: weight decay coefficient (default: 0.01)
- `amsgrad`: whether to use the AMSGrad variant (default: False)
- `maximize`: maximize the params based on the objective, instead of minimizing (default: False)
- `fused`: whether to use the fused implementation (default: None, auto-detect)

**Example**

```python
model = nn.Linear(10, 5)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer.zero_grad()
loss = model(input).sum()
loss.backward()
optimizer.step()
```

### `CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L12)

Callback to save model checkpoints at specified intervals.

**Parameters**

- `checkpoint_dir`: Directory to save checkpoints.
- `save_every`: Save checkpoint every N steps.
- `keep_last_n`: Keep only the last N checkpoints.
- `save_best`: Whether to save the best model based on validation loss.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L36)

Save checkpoint after each optimization step if needed.

### `CosineAnnealingLR(optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/cosine.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L7)

Cosine annealing learning rate scheduler.

Decreases the learning rate following a cosine curve from the initial
learning rate to eta_min over T_max steps.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `T_max`: Maximum number of iterations (steps).
- `eta_min`: Minimum learning rate (default: 0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/cosine.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/cosine.py#L39)

Compute learning rate using cosine annealing.

### `CrossEntropyLoss(reduction='mean') -> None`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/cross_entropy.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L6)

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/train/losses/cross_entropy.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L8)

### `DDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, ddp_backend: str | None = None, find_unused_parameters: bool = False, broadcast_buffers: bool = True, bucket_cap_mb: int = 25, gradient_as_bucket_view: bool = True, static_graph: bool = False)`

**Bases:** `olm.train.trainer.trainer.Trainer`

Source: [`src/olm/train/trainer/ddp_trainer.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/ddp_trainer.py#L27)

Trainer with PyTorch Distributed Data Parallel (DDP) support.

Wraps the model with torch.nn.parallel.DistributedDataParallel and handles:
- Distributed sampler setup
- Gradient synchronization (with no_sync for gradient accumulation)
- Metrics aggregation across ranks
- Checkpoint saving on rank 0

**Parameters**

- `model`: Model to train.
- `optimizer`: Optimizer instance or class.
- `dataloader`: DataLoader (will add DistributedSampler if needed).
- `device`: Device for training.
- `context_length`: Max sequence length.
- `grad_accum_steps`: Gradient accumulation steps.
- `use_amp`: Use automatic mixed precision.
- `loss`: Loss function class.
- `callbacks`: Training callbacks.
- `scheduler`: Learning rate scheduler.
- `grad_clip_norm`: Gradient clipping threshold.
- `warmup_steps`: Warmup steps for scheduler.
- `total_steps`: Total training steps.
- `min_lr`: Minimum learning rate.
- `learning_rate`: Learning rate (if optimizer is a class).
- `weight_decay`: Weight decay (if optimizer is a class).
- `use_warmup_cosine`: Use warmup+cosine scheduler by default.
- `ddp_backend`: DDP backend ('nccl' for GPU, 'gloo' for CPU, None for auto).
- `find_unused_parameters`: DDP parameter for models with unused params.
- `broadcast_buffers`: Broadcast model buffers at beginning of forward.
- `bucket_cap_mb`: DDP bucket size in MB for gradient communication.
- `gradient_as_bucket_view`: Use gradient views to reduce memory (recommended).
- `static_graph`: Set to True if model graph doesn't change (optimization).

**Example**

```python
# Launch with: torchrun --nproc_per_node=4 train.py
from olm.core.dist import setup_distributed
setup_distributed()

trainer = DDPTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=dataloader,
    device=f"cuda:{get_local_rank()}",
    context_length=512,
    learning_rate=3e-4
)
trainer.train(epochs=10)
```

#### Methods

##### `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`

Source: [`src/olm/train/trainer/ddp_trainer.py:150`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/ddp_trainer.py#L150)

Training loop with DDP support.

**Parameters**

- `epochs`: Number of epochs.
- `log_interval`: Log every N steps.
- `max_steps`: Maximum steps to train.
- `steps_per_epoch`: Max steps per epoch.

**Returns**

List of loss values (only on rank 0).

### `DeviceConfig(num_gpus: int, num_cpus: int, cuda_available: bool, gpu_memory_per_device: float | None = None, total_gpu_memory: float | None = None, strategy: olm.train.device.TrainerStrategy | None = None, device_type: str = 'cuda', local_rank: int = 0, world_size: int = 1, backend: str | None = None, mixed_precision: str | None = None, sharding_strategy: str | None = None, auto_wrap_policy: str | None = None, cpu_offload: bool = False) -> None`

Source: [`src/olm/train/device.py:27`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L27)

Configuration for device and training strategy.

**Attributes**

- `num_gpus`: Number of available GPUs
- `num_cpus`: Number of CPU cores
- `cuda_available`: Whether CUDA is available
- `gpu_memory_per_device`: GPU memory in GB per device
- `total_gpu_memory`: Total GPU memory in GB
- `strategy`: Selected training strategy
- `device_type`: Device type ('cuda' or 'cpu')
- `local_rank`: Local rank for distributed training
- `world_size`: World size for distributed training
- `backend`: Distributed backend ('nccl', 'gloo', or None)
- `mixed_precision`: Mixed precision dtype ('bf16', 'fp16', or None)
- `sharding_strategy`: FSDP sharding strategy (if applicable)
- `auto_wrap_policy`: FSDP auto wrap policy (if applicable)
- `cpu_offload`: Whether to offload parameters to CPU

#### Methods

##### `to_dict(self) -> Dict[str, Any]`

Source: [`src/olm/train/device.py:64`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L64)

Convert config to dictionary.

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L8)

Callback to stop training early if validation loss doesn't improve.

**Parameters**

- `patience`: Number of validation checks to wait for improvement.
- `min_delta`: Minimum change in validation loss to qualify as improvement.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L24)

Check for early stopping after each step.

### `FSDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, sharding_strategy: str = 'FULL_SHARD', auto_wrap_policy: str | None = 'size', min_num_params: int = 100000000, transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, cpu_offload: bool = False, backward_prefetch: str = 'BACKWARD_PRE', mixed_precision_policy: str | None = None, limit_all_gathers: bool = True, use_orig_params: bool = True)`

**Bases:** `olm.train.trainer.trainer.Trainer`

Source: [`src/olm/train/trainer/fsdp_trainer.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L42)

Trainer with PyTorch Fully Sharded Data Parallel (FSDP) support.

FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of larger models than DDP. Uses PyTorch's native FSDP.

**Parameters**

- `model`: Model to train.
- `optimizer`: Optimizer instance or class.
- `dataloader`: DataLoader (will add DistributedSampler if needed).
- `device`: Device for training.
- `context_length`: Max sequence length.
- `grad_accum_steps`: Gradient accumulation steps.
- `use_amp`: Use automatic mixed precision.
- `loss`: Loss function class.
- `callbacks`: Training callbacks.
- `scheduler`: Learning rate scheduler.
- `grad_clip_norm`: Gradient clipping threshold.
- `warmup_steps`: Warmup steps for scheduler.
- `total_steps`: Total training steps.
- `min_lr`: Minimum learning rate.
- `learning_rate`: Learning rate (if optimizer is a class).
- `weight_decay`: Weight decay (if optimizer is a class).
- `use_warmup_cosine`: Use warmup+cosine scheduler by default.
- `sharding_strategy`: FSDP sharding strategy: - FULL_SHARD: Shard parameters, gradients, optimizer states (most memory efficient) - SHARD_GRAD_OP: Shard gradients and optimizer states only - NO_SHARD: Equivalent to DDP - HYBRID_SHARD: Full shard within node, replicate across nodes
- `auto_wrap_policy`: Policy for automatic module wrapping: - "size": Wrap based on parameter count (default, uses min_num_params) - "transformer": Wrap transformer layers (provide transformer_layer_cls) - None: Manual wrapping (model must already be wrapped)
- `min_num_params`: Minimum parameters for size-based wrapping (default: 1e8 = 100M).
- `transformer_layer_cls`: Transformer layer class for transformer wrapping policy.
- `cpu_offload`: Offload parameters to CPU when not in use.
- `backward_prefetch`: Prefetch parameters for backward pass (recommended).
- `mixed_precision_policy`: Mixed precision configuration (BF16, FP16, or None).
- `limit_all_gathers`: Limit all-gather operations for memory efficiency.
- `use_orig_params`: Use original parameters instead of flattened (better for optimizers).

**Example**

```python
# Launch with: torchrun --nproc_per_node=8 train.py
from olm.core.dist import setup_distributed
setup_distributed()

trainer = FSDPTrainer(
    model=model,
    optimizer=torch.optim.AdamW,
    dataloader=dataloader,
    device=f"cuda:{get_local_rank()}",
    context_length=2048,
    learning_rate=3e-4,
    sharding_strategy="FULL_SHARD",
    auto_wrap_policy="size",
    min_num_params=1e8,  # Wrap layers with 100M+ params
    mixed_precision_policy="bf16"
)
trainer.train(epochs=10)
```

#### Methods

##### `save_checkpoint(self, path: str, state_dict_type: str = 'FULL_STATE_DICT') -> None`

Source: [`src/olm/train/trainer/fsdp_trainer.py:516`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L516)

Save FSDP checkpoint.

**Parameters**

- `path`: Path to save checkpoint.
- `state_dict_type`: Type of state dict to save: - "FULL_STATE_DICT": Gather full model on rank 0 (recommended) - "LOCAL_STATE_DICT": Save local shards on each rank - "SHARDED_STATE_DICT": Save sharded checkpoint

##### `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`

Source: [`src/olm/train/trainer/fsdp_trainer.py:240`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L240)

Training loop with FSDP support.

**Parameters**

- `epochs`: Number of epochs.
- `log_interval`: Log every N steps.
- `max_steps`: Maximum steps to train.
- `steps_per_epoch`: Max steps per epoch.

**Returns**

List of loss values (only on rank 0).

### `KLLoss(kl_coeff=1.0, from_logits=True, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/kllloss.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/kllloss.py#L7)

Forward KL penalty between a policy distribution and a reference distribution.

Supports two input modes:
  1) From logits:
     logits      : [B, T, V]  (policy logits)
     ref  : [B, T, V]  (reference logits)

  2) From log-probs:
     logp       : [B, T, V]  (policy log-probs)
     ref_logp    : [B, T, V]  (reference log-probs)

Optional:
  loss_mask : [B, T] (bool or 0/1) to mask tokens

Output:
  scalar loss (unless reduction="none") = kl_coeff * reduced(KL per token)

Notes:
  - This computes the *full-distribution* KL per token (sums over vocab V).
  - That's the common KL regularizer used to keep the policy close to a reference.

#### Methods

##### `forward(self, logits, ref, mask=None)`

Source: [`src/olm/train/losses/kllloss.py:57`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/kllloss.py#L57)

### `LRMonitorCallback(log_every: int = 100)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L8)

Callback to monitor and log learning rate.

**Parameters**

- `log_every`: Log learning rate every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/lr_monitor_cb.py:19`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/lr_monitor_cb.py#L19)

Log learning rate after each optimization step if needed.

### `LinearDecayLR(optimizer, total_steps: int, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/linear.py:66`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L66)

Simple linear decay scheduler that decays to zero.

This is a simplified version that always decays to 0 from the initial LR.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `total_steps`: Total number of steps to decay over.
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import LinearDecayLR
scheduler = LinearDecayLR(optimizer, total_steps=1000)
for step in range(total_steps):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/linear.py:89`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L89)

Compute learning rate using linear decay.

### `LinearLR(optimizer, total_steps: int, end_lr: float = 0, start_factor: float = 1.0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/linear.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L6)

Linear learning rate scheduler.

Linearly decreases (or increases) the learning rate from the initial
learning rate to end_lr over total_steps.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `total_steps`: Total number of steps for the schedule.
- `end_lr`: Target learning rate at the end (default: 0).
- `start_factor`: Initial learning rate multiplier (default: 1.0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import LinearLR
# Decay from initial LR to 0
scheduler = LinearLR(optimizer, total_steps=1000, end_lr=0)
for step in range(total_steps):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/linear.py:42`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/linear.py#L42)

Compute learning rate using linear interpolation.

### `Lion(params: Iterable, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False)`

**Bases:** `olm.train.optim.base.OptimizerBase`

Source: [`src/olm/train/optim/lion.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/lion.py#L7)

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from "Symbolic Discovery of Optimization Algorithms"
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

**Parameters**

- `params`: iterable of parameters to optimize or dicts defining parameter groups
- `lr`: learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
- `betas`: coefficients used for computing running averages (default: (0.9, 0.99))
- `weight_decay`: weight decay coefficient (default: 0.0)
- `use_triton`: whether to use Triton kernel for faster computation (default: False)

**Example**

```python
model = nn.Linear(10, 5)
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
optimizer.zero_grad()
loss = model(input).sum()
loss.backward()
optimizer.step()
```

#### Methods

##### `zero_grad(self, set_to_none: bool = True)`

Source: [`src/olm/train/optim/lion.py:126`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/lion.py#L126)

Sets gradients of all optimized tensors to zero.

**Parameters**

- `set_to_none`: instead of setting to zero, set the grads to None. This is more memory efficient and can slightly improve performance.

### `LossBase(reduction='mean') -> None`

**Bases:** `Module`, `ABC`

Source: [`src/olm/train/losses/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L8)

Base class for all loss modules.

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor`

Source: [`src/olm/train/losses/base.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L32)

Apply loss to ``logits`` and ``y``.

### `MaskedCELoss(ignore_index=-100, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/mce.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/mce.py#L5)

Token-level cross-entropy with optional loss_mask.

Expects:
  batch["logits"] : [B, T, V]
  batch["labels"] : [B, T]  (use ignore_index for tokens to ignore)
Optional:
  batch["loss_mask"] : [B, T] (1/0 or bool)

#### Methods

##### `forward(self, logits, y, mask)`

Source: [`src/olm/train/losses/mce.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/mce.py#L22)

### `MetricsLoggerCallback(log_dir: str = 'logs', log_every: int = 10)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L10)

Callback to log metrics to a JSONL file.

**Parameters**

- `log_dir`: Directory to save logs.
- `log_every`: Log metrics every N steps.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/metrics_logger_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/metrics_logger_cb.py#L30)

Log metrics after each optimization step if needed.

### `OptimizerBase(params: collections.abc.Iterable[torch.Tensor] | collections.abc.Iterable[dict[str, Any]] | collections.abc.Iterable[tuple[str, torch.Tensor]], defaults: dict[str, typing.Any]) -> None`

**Bases:** `Optimizer`, `ABC`

Source: [`src/olm/train/optim/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L8)

Abstract base class for all optimizers in the OLM framework.

Provides a consistent interface for optimizer implementations, including
standard methods for parameter updates, gradient zeroing, and state management.
All custom optimizers should inherit from this class.

This base class extends PyTorch's Optimizer class and adds additional
functionality specific to the OLM framework.

Subclasses must implement the step() method to define the optimization logic.

#### Methods

##### `extra_repr(self) -> str`

Source: [`src/olm/train/optim/base.py:74`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L74)

String representation of the optimizer for debugging.

Override this in subclasses to provide useful information.

##### `load_state_dict(self, state_dict: Dict[str, Any])`

Source: [`src/olm/train/optim/base.py:64`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L64)

Loads the optimizer state.

**Parameters**

- `state_dict`: optimizer state. Should be an object returned from a call to state_dict().

##### `state_dict(self) -> Dict[str, Any]`

Source: [`src/olm/train/optim/base.py:48`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L48)

Returns the state of the optimizer as a dict.

It contains two entries:

- ``state``: dict holding current optimization state. Its content
  differs between optimizer classes.
- ``param_groups``: list containing all parameter groups where each
  parameter group is a dict.

**Returns**

Dictionary containing optimizer state

##### `step(self, closure: Callable[[], float] | None = None) -> float | None`

Source: [`src/olm/train/optim/base.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L22)

Performs a single optimization step.

**Parameters**

- `closure`: A closure that reevaluates the model and returns the loss. Some optimization algorithms (e.g., L-BFGS) require multiple evaluations of the loss function.

**Returns**

Optional loss value if closure is provided.

##### `zero_grad(self, set_to_none: bool = True)`

Source: [`src/olm/train/optim/base.py:37`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/base.py#L37)

Sets gradients of all optimized tensors to zero or None.

**Parameters**

- `set_to_none`: Instead of setting to zero, set the grads to None. This is more memory efficient and can slightly improve performance.
- `Default`: True

### `SchedulerBase(optimizer, last_epoch: int = -1, verbose: bool = False)`

**Bases:** `_LRScheduler`, `ABC`

Source: [`src/olm/train/schedulers/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L8)

Base class for all OLM learning rate schedulers.

This class extends PyTorch's _LRScheduler and provides a consistent
interface for implementing custom learning rate schedules. All OLM
schedulers should inherit from this class to maintain uniformity.

Subclasses must implement:
    - get_lr(): Compute the learning rate for the current step
    - _get_closed_form_lr() (optional): Closed-form solution for efficiency

**Parameters**

- `optimizer`: Wrapped PyTorch optimizer.
- `last_epoch`: The index of the last epoch (default: -1).
- `verbose`: If True, prints a message to stdout for each update (default: False).

**Example**

```python
class MyScheduler(SchedulerBase):
    def __init__(self, optimizer, param, last_epoch=-1):
        self.param = param
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Custom logic here
        return [base_lr * self.param for base_lr in self.base_lrs]
```

#### Methods

##### `get_last_lr(self) -> List[float]`

Source: [`src/olm/train/schedulers/base.py:64`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L64)

Return last computed learning rate by current scheduler.

**Returns**

List of last computed learning rates.

##### `get_lr(self) -> List[float]`

Source: [`src/olm/train/schedulers/base.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L39)

Compute learning rate for each parameter group.

This method must be implemented by subclasses to define the
learning rate schedule logic.

**Returns**

List of learning rates, one per parameter group.

##### `load_state_dict(self, state_dict)`

Source: [`src/olm/train/schedulers/base.py:86`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L86)

Load the scheduler state from a checkpoint.

**Parameters**

- `state_dict`: Scheduler state returned by state_dict().

##### `state_dict(self)`

Source: [`src/olm/train/schedulers/base.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/base.py#L73)

Returns the state of the scheduler as a dict.

Contains all non-callable attributes that are specific to
the scheduler and required for checkpointing.

### `ThroughputCallback(log_every: int = 100, context_length: int = 1024, batch_size: int = 8)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/throughput_cb.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L9)

Callback to monitor training throughput (tokens/sec, samples/sec).

**Parameters**

- `log_every`: Log throughput every N steps.
- `context_length`: Length of each sequence.
- `batch_size`: Total batch size (including gradient accumulation).

#### Methods

##### `on_step_begin(self, trainer, step: int) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:30`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L30)

Record start time of the step.

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/throughput_cb.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/throughput_cb.py#L34)

Calculate and log throughput.

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

**Attributes**

- `model` (`Pipeline`): The model to train.
- `optimizer` (`torch.optim.Optimizer`): The optimizer to use.
- `dataloader` (`olm.data.datasets.data_loader.DataLoader`): The data provider.
- `device` (`str`): The device to train on (e.g., 'cuda', 'cpu').
- `context_length` (`int`): The maximum sequence length for training.
- `grad_accum_steps` (`int`): Number of steps to accumulate gradients before updating.
- `use_amp` (`bool`): Whether to use Automatic Mixed Precision.
- `scaler` (`GradScaler`): Gradient scaler for AMP.
- `loss` (`LossBase`): The loss function instance.
- `callbacks` (`List[TrainerCallback]`): List of callbacks to execute during training.
- `scheduler` (`Optional`): Learning rate scheduler to step after each optimization step.
- `global_step` (`int`): Current global step count.
- `current_epoch` (`int`): Current epoch number.
- `total_tokens_processed` (`int`): Total number of tokens processed during training.
- `step_start_time` (`float`): Timestamp of the current step start.
- `training_start_time` (`float`): Timestamp when training began.

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

**Parameters**

- `epochs` (`int`): The number of complete passes through the dataset.
- `log_interval` (`int`): How often to print the loss. Defaults to 10.
- `max_steps` (`int, optional`): Maximum number of steps to train for.
- `steps_per_epoch` (`int, optional`): Maximum number of steps per epoch. Defaults to None (unlimited).

**Returns**

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

### `TrainerStrategy(*values)`

**Bases:** `Enum`

Source: [`src/olm/train/device.py:17`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/device.py#L17)

Training strategy based on available hardware.

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/validation_cb.py:11`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L11)

Callback to perform validation at specified intervals.

**Parameters**

- `val_dataloader`: Validation dataloader.
- `eval_every`: Validate every N steps.
- `device`: Device to run validation on.
- `use_amp`: Whether to use automatic mixed precision.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/validation_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L36)

Run validation after each optimization step if needed.

### `WarmupCosineScheduler(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/warmup.py:62`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L62)

Combined warmup and cosine annealing scheduler.

Linearly warms up the learning rate from 0 to base_lr over warmup_steps,
then applies cosine annealing decay to min_lr over the remaining steps.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `warmup_steps`: Number of warmup steps.
- `total_steps`: Total number of training steps.
- `min_lr`: Minimum learning rate after decay (default: 0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import WarmupCosineScheduler
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    min_lr=1e-6
)
for step in range(total_steps):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/warmup.py:102`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L102)

Compute learning rate with warmup and cosine decay.

### `WarmupLR(optimizer, warmup_steps: int, start_lr: float = 0, last_epoch: int = -1)`

**Bases:** `olm.train.schedulers.base.SchedulerBase`

Source: [`src/olm/train/schedulers/warmup.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L6)

Learning rate warmup scheduler.

Linearly increases the learning rate from 0 to the base learning rate
over warmup_steps.

**Parameters**

- `optimizer`: Wrapped optimizer.
- `warmup_steps`: Number of warmup steps.
- `start_lr`: Initial learning rate (default: 0).
- `last_epoch`: The index of last epoch (default: -1).

**Example**

```python
from olm.train.schedulers import WarmupLR
scheduler = WarmupLR(optimizer, warmup_steps=1000)
for step in range(warmup_steps):
    train(...)
    scheduler.step()
```

#### Methods

##### `get_lr(self)`

Source: [`src/olm/train/schedulers/warmup.py:38`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/schedulers/warmup.py#L38)

Compute learning rate during warmup.

### `ZLoss(z_coeff=0.0001, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/zloss.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/zloss.py#L4)

Z-loss (logZ^2 penalty), commonly used as an auxiliary regularizer.

For each token:
  logZ = logsumexp(logits, dim=-1)
  zloss = (logZ ** 2)

Notes:
- This does NOT include CE. Usually you add it to CE:
    total = ce_loss + z_coeff * z_loss

#### Methods

##### `forward(self, logits, y, mask=None)`

Source: [`src/olm/train/losses/zloss.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/zloss.py#L24)
