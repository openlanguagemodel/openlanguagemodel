"""
Fully Sharded Data Parallel (FSDP) Trainer using PyTorch's native FSDP.
"""

from pathlib import Path
from typing import Optional, List, Type, Union, Any, Callable, Dict
import os
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

from olm.train.trainer.ddp_trainer import _DataParallelMixin
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
    print_rank_0,
)
from olm.train.checkpointing import write_checkpoint


class FSDPTrainer(_DataParallelMixin, Trainer):
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
        resume_from: Only ``None`` is supported for a sharded model; a checkpoint
            raises ``NotImplementedError`` (see ``load_state_dict``). ``"latest"``
            or a directory with no checkpoint in it starts a new run.

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
        resume_from: Optional[Union[str, os.PathLike]] = None,
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
            resume_from=resume_from,
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

        return super().train(
            epochs,
            log_interval=log_interval,
            max_steps=max_steps,
            steps_per_epoch=steps_per_epoch,
        )

    def _clip_gradients(self) -> None:
        """Clip with FSDP's own method, which computes the norm over all shards."""
        if isinstance(self.model, FSDP):
            self.model.clip_grad_norm_(self.grad_clip_norm)
        else:
            super()._clip_gradients()

    def state_dict(self) -> Dict[str, Any]:
        """
        Not available for a sharded model; use :meth:`save_checkpoint`.

        Gathering FSDP parameters is a collective operation that every rank
        must join, which a single-process call cannot guarantee.
        """
        if not isinstance(self.model, FSDP):
            return super().state_dict()
        raise NotImplementedError(
            "FSDPTrainer.state_dict() is not available for a sharded model. Call "
            "save_checkpoint(path) on every rank instead."
        )

    def load_state_dict(self, checkpoint: Dict[str, Any], strict: bool = True) -> None:
        """
        Resuming a sharded run is not supported yet.

        To start a new FSDP run from saved weights, load them into the model
        before creating the trainer::

            model.load_state_dict(read_checkpoint(path)["model"])
            trainer = FSDPTrainer(model, ...)
        """
        if not isinstance(self.model, FSDP):
            return super().load_state_dict(checkpoint, strict=strict)
        raise NotImplementedError(
            "FSDPTrainer cannot resume a sharded run yet: its checkpoints hold the "
            "full model weights but no optimizer state. To start a new run from "
            "those weights, call model.load_state_dict(read_checkpoint(path)"
            '["model"]) before creating the FSDPTrainer. See '
            "docs/guides/checkpointing.md."
        )

    def save_checkpoint(
        self,
        path: str,
        state_dict_type: str = "FULL_STATE_DICT",
    ) -> Optional[Path]:
        """
        Save FSDP checkpoint.

        All ranks must call this. With ``"FULL_STATE_DICT"`` the full model is
        gathered on rank 0, which writes a format version 1 checkpoint (see
        ``olm.train.checkpointing``) without optimizer state: the sharded
        optimizer state is not gathered yet, so these files are for weights,
        evaluation and inspection, not for resuming.

        Args:
            path: Path to save checkpoint.
            state_dict_type: Type of state dict to save:
                - "FULL_STATE_DICT": Gather full model on rank 0 (recommended)
                - "LOCAL_STATE_DICT": Save local shards on each rank
                - "SHARDED_STATE_DICT": Save sharded checkpoint

        Returns:
            The written path on processes that wrote a file, else ``None``.
        """
        if not isinstance(self.model, FSDP):
            return super().save_checkpoint(path)

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
        rng_states = self._gather_rng_states() if is_full else None
        written = None

        if is_full and is_main_process():
            checkpoint = self._build_checkpoint(state_dict, None, rng_states)
            written = write_checkpoint(checkpoint, path)
        elif not is_full:
            checkpoint_path = Path(path)
            checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.rank{get_rank()}{checkpoint_path.suffix}"
            )
            checkpoint = {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.scheduler.state_dict() if self.scheduler else None),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
            torch.save(checkpoint, checkpoint_path)
            written = checkpoint_path

        if is_distributed():
            barrier()
        return written
