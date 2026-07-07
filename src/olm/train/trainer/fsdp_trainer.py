"""
Fully Sharded Data Parallel (FSDP) Trainer using PyTorch's native FSDP.
"""

from pathlib import Path
from typing import Optional, List, Type, Union, Any, Callable
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    LocalStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.data import DistributedSampler
import functools

from olm.train.trainer.trainer import Trainer, TrainerCallback
from olm.train.losses.base import LossBase
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.data.datasets import DataLoader
from olm.core.dist import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    all_reduce,
    print_rank_0,
)


class FSDPTrainer(Trainer):
    """
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
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer]],
        dataloader: DataLoader,
        device: str,
        context_length: int,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        loss: Type[LossBase] = CrossEntropyLoss,
        mtp_loss: Optional[Union[LossBase, Type[LossBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        scheduler: Optional[Any] = None,
        grad_clip_norm: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        min_lr: float = 0.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        use_warmup_cosine: bool = True,
        # FSDP-specific parameters
        sharding_strategy: str = "FULL_SHARD",
        auto_wrap_policy: Optional[str] = "size",
        min_num_params: int = int(1e8),
        transformer_layer_cls: Optional[Type[nn.Module]] = None,
        cpu_offload: bool = False,
        backward_prefetch: str = "BACKWARD_PRE",
        mixed_precision_policy: Optional[str] = None,
        limit_all_gathers: bool = True,
        use_orig_params: bool = True,
    ):
        self.sharding_strategy_name = sharding_strategy
        self.auto_wrap_policy_type = auto_wrap_policy
        self.min_num_params = min_num_params
        self.transformer_layer_cls = transformer_layer_cls
        self.cpu_offload_enabled = cpu_offload
        self.backward_prefetch_name = backward_prefetch
        self.mixed_precision_policy_name = mixed_precision_policy
        self.limit_all_gathers = limit_all_gathers
        self.use_orig_params = use_orig_params

        # Move model to device first (before FSDP wrapping)
        model = model.to(device)

        # Wrap model with FSDP before calling super().__init__
        if is_distributed():
            model = self._wrap_model_fsdp(model, device)
            print_rank_0(f"Model wrapped with FSDP (sharding: {sharding_strategy})")

        # Initialize base trainer
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            context_length=context_length,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            loss=loss,
            mtp_loss=mtp_loss,
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

    def _wrap_model_fsdp(self, model: nn.Module, device: str) -> FSDP:
        """Wrap model with FSDP."""
        # Parse sharding strategy
        sharding_strategy = getattr(ShardingStrategy, self.sharding_strategy_name)

        # Create auto wrap policy
        auto_wrap_policy = None
        if self.auto_wrap_policy_type == "size":
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=self.min_num_params,
            )
        elif self.auto_wrap_policy_type == "transformer":
            if self.transformer_layer_cls is None:
                raise ValueError(
                    "transformer_layer_cls must be provided for transformer wrap policy"
                )
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={self.transformer_layer_cls},
            )

        # CPU offload
        cpu_offload = (
            CPUOffload(offload_params=True) if self.cpu_offload_enabled else None
        )

        # Backward prefetch
        backward_prefetch = (
            getattr(BackwardPrefetch, self.backward_prefetch_name)
            if self.backward_prefetch_name
            else None
        )

        # Mixed precision
        mixed_precision = None
        if self.mixed_precision_policy_name:
            if self.mixed_precision_policy_name.lower() == "bf16":
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            elif self.mixed_precision_policy_name.lower() == "fp16":
                mixed_precision = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            device_id=(
                torch.cuda.current_device() if torch.cuda.is_available() else None
            ),
            limit_all_gathers=self.limit_all_gathers,
            use_orig_params=self.use_orig_params,
        )

        return fsdp_model

    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        max_steps: int = None,
        steps_per_epoch: int = None,
    ) -> list[float]:
        """
        Training loop with FSDP support.

        Args:
            epochs: Number of epochs.
            log_interval: Log every N steps.
            max_steps: Maximum steps to train.
            steps_per_epoch: Max steps per epoch.

        Returns:
            List of loss values (only on rank 0).
        """
        # Setup distributed sampler if needed
        if is_distributed() and hasattr(self.dataloader, "sampler"):
            if not isinstance(self.dataloader.sampler, DistributedSampler):
                print_rank_0(
                    "Warning: DataLoader not using DistributedSampler. "
                    "Data may not be properly distributed across ranks."
                )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Initialize scheduler
        if self.scheduler is None and self.use_warmup_cosine:
            self._initialize_scheduler(epochs, max_steps, steps_per_epoch)

        losses = []

        def finish_accumulation(actual_count: int) -> bool:
            nonlocal accumulated_loss, accumulated_tokens, accumulation_count, epoch_step

            self._call_callbacks("on_step_begin", self, self.global_step)

            needs_unscale = (
                self.grad_clip_norm is not None or actual_count != accumulation_target
            )
            if needs_unscale:
                self.scaler.unscale_(self.optimizer)

            if actual_count != accumulation_target:
                grad_scale = accumulation_target / actual_count
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.mul_(grad_scale)

            if self.grad_clip_norm is not None:
                self.model.clip_grad_norm_(self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = accumulated_loss / actual_count
            self.global_step += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            perplexity = math.exp(min(avg_loss, 20))

            step_time = (
                time.time() - self.step_start_time if self.step_start_time else 0.0
            )
            tokens_per_sec = accumulated_tokens / step_time if step_time > 0 else 0.0

            if is_distributed():
                loss_tensor = torch.tensor([avg_loss], device=self.device)
                all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
                avg_loss = loss_tensor.item()
                perplexity = math.exp(min(avg_loss, 20))

                throughput_tensor = torch.tensor([tokens_per_sec], device=self.device)
                all_reduce(throughput_tensor, op=torch.distributed.ReduceOp.SUM)
                tokens_per_sec = throughput_tensor.item()

            self.training_state.update(
                {
                    "current_loss": avg_loss,
                    "perplexity": perplexity,
                    "tokens_per_sec": tokens_per_sec,
                    "learning_rate": current_lr,
                    "total_tokens": self.total_tokens_processed,
                }
            )

            if is_main_process():
                losses.append(avg_loss)
                self.losses.append(avg_loss)

            if self.global_step % log_interval == 0 and is_main_process():
                print(
                    f"{epoch+1:^6} | {self.global_step:^8} | {avg_loss:^10.4f} | "
                    f"{perplexity:^11.2f} | {tokens_per_sec:^10.0f} | {current_lr:^10.2e}",
                    flush=True,
                )

            self._call_callbacks("on_step_end", self, self.global_step, avg_loss)
            should_stop = self._should_stop_training()

            accumulated_loss = 0.0
            accumulated_tokens = 0
            accumulation_count = 0
            epoch_step += 1

            return should_stop

        # Call callbacks
        self._call_callbacks("on_train_begin", self)

        import time
        import math

        self.training_start_time = time.time()

        # Print header only on rank 0
        if is_main_process():
            print(
                f"{'Epoch':^6} | {'Step':^8} | {'Loss':^10} | {'Perplexity':^11} | "
                f"{'Tokens/s':^10} | {'LR':^10}",
                flush=True,
            )
            print("-" * 80, flush=True)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Set epoch for DistributedSampler
            if is_distributed() and hasattr(self.dataloader, "sampler"):
                if isinstance(self.dataloader.sampler, DistributedSampler):
                    self.dataloader.sampler.set_epoch(epoch)

            self._call_callbacks("on_epoch_begin", self, epoch)

            accumulated_loss = 0.0
            accumulated_tokens = 0
            accumulation_count = 0
            accumulation_target = self.grad_accum_steps
            epoch_step = 0
            try:
                num_batches = len(self.dataloader)
            except TypeError:
                num_batches = None

            for step, (x, y) in enumerate(self.dataloader):
                self._call_callbacks("on_batch_begin", self, step)

                # Start timing
                if accumulation_count == 0:
                    self.step_start_time = time.time()
                    if num_batches is None:
                        accumulation_target = self.grad_accum_steps
                    else:
                        accumulation_target = min(
                            self.grad_accum_steps, num_batches - step
                        )

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # Track tokens
                batch_tokens = x.numel()
                self.total_tokens_processed += batch_tokens
                accumulated_tokens += batch_tokens

                # FSDP automatically handles gradient synchronization
                # For gradient accumulation, we still accumulate normally
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    model_output = self.model(x)
                    loss, _ = self._compute_model_loss(model_output, y)
                    loss_val = loss.item()
                    loss = loss / accumulation_target

                self.scaler.scale(loss).backward()

                accumulated_loss += loss_val
                accumulation_count += 1
                self.training_state["current_loss"] = loss_val
                self.training_state["accumulated_loss"] = accumulated_loss

                self._call_callbacks("on_batch_end", self, step, loss_val)

                # Optimizer step after gradient accumulation
                if accumulation_count == accumulation_target:
                    if finish_accumulation(accumulation_count):
                        self._call_callbacks("on_epoch_end", self, epoch)
                        self._call_callbacks("on_train_end", self)
                        if is_main_process():
                            self.losses = losses
                            self._print_training_summary()
                        return losses

                    # Check stopping conditions
                    if max_steps and self.global_step >= max_steps:
                        self._call_callbacks("on_epoch_end", self, epoch)
                        self._call_callbacks("on_train_end", self)
                        if is_main_process():
                            self.losses = losses
                            self._print_training_summary()
                        return losses

                    if steps_per_epoch and epoch_step >= steps_per_epoch:
                        break

            if accumulation_count > 0:
                if finish_accumulation(accumulation_count):
                    self._call_callbacks("on_epoch_end", self, epoch)
                    self._call_callbacks("on_train_end", self)
                    if is_main_process():
                        self.losses = losses
                        self._print_training_summary()
                    return losses

                if max_steps and self.global_step >= max_steps:
                    self._call_callbacks("on_epoch_end", self, epoch)
                    self._call_callbacks("on_train_end", self)
                    if is_main_process():
                        self.losses = losses
                        self._print_training_summary()
                    return losses

            self._call_callbacks("on_epoch_end", self, epoch)

        self._call_callbacks("on_train_end", self)
        if is_main_process():
            self._print_training_summary()

        self.losses = losses
        return losses

    def _initialize_scheduler(self, epochs, max_steps, steps_per_epoch):
        """Initialize default warmup+cosine scheduler."""
        from olm.train.schedulers.warmup import WarmupCosineScheduler

        # Calculate total steps
        if self.total_steps is None:
            if max_steps is not None:
                self.total_steps = max_steps
            elif steps_per_epoch is not None:
                self.total_steps = epochs * steps_per_epoch
            else:
                try:
                    dataset_size = len(self.dataloader)
                    self.total_steps = epochs * dataset_size
                except TypeError:
                    self.total_steps = epochs * 10000

        # Calculate warmup steps
        if self.warmup_steps is None:
            self.warmup_steps = min(
                max(int(0.1 * self.total_steps), 1),
                5000,
                self.total_steps,
            )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr=self.min_lr,
        )

        print_rank_0(
            f"Initialized WarmupCosineScheduler: warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, min_lr={self.min_lr}"
        )

    def save_checkpoint(
        self,
        path: str,
        state_dict_type: str = "FULL_STATE_DICT",
    ) -> None:
        """
        Save FSDP checkpoint.

        Args:
            path: Path to save checkpoint.
            state_dict_type: Type of state dict to save:
                - "FULL_STATE_DICT": Gather full model on rank 0 (recommended)
                - "LOCAL_STATE_DICT": Save local shards on each rank
                - "SHARDED_STATE_DICT": Save sharded checkpoint
        """
        state_dict_type_enum = getattr(StateDictType, state_dict_type)
        if state_dict_type == "FULL_STATE_DICT":
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
        elif state_dict_type == "LOCAL_STATE_DICT":
            state_dict_config = LocalStateDictConfig()
        elif state_dict_type == "SHARDED_STATE_DICT":
            state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        else:
            raise ValueError(f"Unsupported FSDP state_dict_type: {state_dict_type}")

        # All ranks must participate in the state dict collective (especially for
        # FULL_STATE_DICT, which gathers shards across the process group).
        with FSDP.state_dict_type(self.model, state_dict_type_enum, state_dict_config):
            state_dict = self.model.state_dict()

            is_full = state_dict_type == "FULL_STATE_DICT"
            should_write = is_main_process() if is_full else True

            if should_write:
                checkpoint_path = Path(path)
                if not is_full:
                    checkpoint_path = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}.rank{get_rank()}{checkpoint_path.suffix}"
                    )

                checkpoint = {
                    "model": state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.scheduler.state_dict() if self.scheduler else None
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        if is_distributed():
            barrier()
