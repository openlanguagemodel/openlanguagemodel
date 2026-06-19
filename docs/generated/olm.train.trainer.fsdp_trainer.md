# `olm.train.trainer.fsdp_trainer`

Source: [`src/olm/train/trainer/fsdp_trainer.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L1)

Fully Sharded Data Parallel (FSDP) Trainer using PyTorch's native FSDP.

## Classes

### `FSDPTrainer(model: torch.nn.modules.module.Module, optimizer: torch.optim.optimizer.Optimizer | Type[torch.optim.optimizer.Optimizer], dataloader: olm.data.datasets.data_loader.DataLoader, device: str, context_length: int, grad_accum_steps: int = 1, use_amp: bool = True, loss: Type[olm.train.losses.base.LossBase] = <class 'olm.train.losses.cross_entropy.CrossEntropyLoss'>, callbacks: List[olm.train.trainer.trainer.TrainerCallback] | None = None, scheduler: Any | None = None, grad_clip_norm: float | None = None, warmup_steps: int | None = None, total_steps: int | None = None, min_lr: float = 0.0, learning_rate: float = 0.0003, weight_decay: float = 0.0, use_warmup_cosine: bool = True, sharding_strategy: str = 'FULL_SHARD', auto_wrap_policy: str | None = 'size', min_num_params: int = 100000000, transformer_layer_cls: Type[torch.nn.modules.module.Module] | None = None, cpu_offload: bool = False, backward_prefetch: str = 'BACKWARD_PRE', mixed_precision_policy: str | None = None, limit_all_gathers: bool = True, use_orig_params: bool = True)`

**Bases:** `olm.train.trainer.trainer.Trainer`

Source: [`src/olm/train/trainer/fsdp_trainer.py:38`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L38)

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

##### `save_checkpoint(self, path: str, state_dict_type: str = 'FULL_STATE_DICT') -> None`

Source: [`src/olm/train/trainer/fsdp_trainer.py:497`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L497)

Save FSDP checkpoint.

Args:
    path: Path to save checkpoint.
    state_dict_type: Type of state dict to save:
        - "FULL_STATE_DICT": Gather full model on rank 0 (recommended)
        - "LOCAL_STATE_DICT": Save local shards on each rank
        - "SHARDED_STATE_DICT": Save sharded checkpoint

##### `train(self, epochs: int, log_interval: int = 10, max_steps: int = None, steps_per_epoch: int = None) -> list[float]`

Source: [`src/olm/train/trainer/fsdp_trainer.py:236`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/trainer/fsdp_trainer.py#L236)

Training loop with FSDP support.

Args:
    epochs: Number of epochs.
    log_interval: Log every N steps.
    max_steps: Maximum steps to train.
    steps_per_epoch: Max steps per epoch.

Returns:
    List of loss values (only on rank 0).
