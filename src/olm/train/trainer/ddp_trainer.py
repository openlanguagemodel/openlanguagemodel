"""
Distributed Data Parallel (DDP) Trainer using PyTorch's native DDP.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Optional, List, Type, Union, Any, Dict, Tuple
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from olm.train.trainer.trainer import Trainer, TrainerCallback
from olm.train.losses.base import LossBase
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.data.datasets import DataLoader
from olm.core.dist import (
    is_distributed,
    get_local_rank,
    all_reduce,
    all_gather_object,
    print_rank_0,
)


class _DataParallelMixin:
    """
    Trainer hooks shared by ``DDPTrainer`` and ``FSDPTrainer``.

    Under a process group every rank trains on its own share of the data. These
    hooks give every epoch its own ``DistributedSampler`` shuffle, combine the
    step metrics and the decision to stop over ranks, and collect every rank's
    random state for checkpoints. Without a process group they do what the
    plain ``Trainer`` does. List the mixin before ``Trainer`` in the bases.
    """

    def _prepare_dataloader_for_epoch(self, epoch: int) -> None:
        """Set the epoch on a DistributedSampler so every epoch shuffles differently."""
        if is_distributed() and hasattr(self.dataloader, "sampler"):
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)

    def _reduce_step_metrics(
        self, avg_loss: float, tokens_per_sec: float
    ) -> Tuple[float, float]:
        """Average the loss and sum the throughput over ranks."""
        if not is_distributed():
            return avg_loss, tokens_per_sec

        loss_tensor = torch.tensor([avg_loss], device=self.device)
        all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)

        throughput_tensor = torch.tensor([tokens_per_sec], device=self.device)
        all_reduce(throughput_tensor, op=torch.distributed.ReduceOp.SUM)
        return loss_tensor.item(), throughput_tensor.item()

    def _should_stop_training(self) -> bool:
        """Stop every rank when any rank's callbacks asked to stop."""
        should_stop = super()._should_stop_training()
        if not is_distributed():
            return should_stop

        flag = torch.tensor([float(should_stop)], device=self.device)
        all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def _gather_rng_states(self) -> List[Dict[str, Any]]:
        """Collect every rank's RNG states; every rank must call this."""
        return all_gather_object(self._rng_state())


class DDPTrainer(_DataParallelMixin, Trainer):
    """
    Trainer with PyTorch Distributed Data Parallel (DDP) support.

    Wraps the model with torch.nn.parallel.DistributedDataParallel and handles:
    - Distributed sampler setup
    - Gradient synchronization (with no_sync for gradient accumulation)
    - Metrics aggregation across ranks
    - Checkpoint saving on rank 0 (every rank must call ``save_checkpoint``;
      the file holds every rank's RNG state)

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
        resume_from: Checkpoint file or directory to resume from, or ``"latest"``
            (see ``Trainer``). Every rank loads the same file.

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
        # DDP-specific parameters
        ddp_backend: Optional[str] = None,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
        bucket_cap_mb: int = 25,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        resume_from: Optional[Union[str, os.PathLike]] = None,
    ):
        # Store DDP config before calling super().__init__
        self.ddp_backend = ddp_backend
        self.find_unused_parameters = find_unused_parameters
        self.broadcast_buffers = broadcast_buffers
        self.bucket_cap_mb = bucket_cap_mb
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph

        # Initialize base trainer (moves model to device)
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

        # Wrap model with DDP if distributed
        if is_distributed():
            self.model = DDP(
                self.model,
                device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
                output_device=get_local_rank() if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers,
                bucket_cap_mb=bucket_cap_mb,
                gradient_as_bucket_view=gradient_as_bucket_view,
                static_graph=static_graph,
            )
            print_rank_0(
                f"Model wrapped with DDP (backend: {self.ddp_backend or 'auto'})"
            )

    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        max_steps: int = None,
        steps_per_epoch: int = None,
    ) -> list[float]:
        """
        Training loop with DDP support.

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

    def _accumulation_context(
        self, sync_gradients: bool, num_batches: Optional[int]
    ) -> AbstractContextManager:
        """
        Skip the gradient all-reduce on all but the last micro-batch of a step.

        Only used when the dataloader has a length: for iterable datasets the
        last micro-batch of an epoch is not known in advance, so every
        micro-batch synchronizes.
        """
        if is_distributed() and num_batches is not None and not sync_gradients:
            return self.model.no_sync()
        return nullcontext()
