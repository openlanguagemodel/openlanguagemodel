"""
Distributed Data Parallel (DDP) Trainer using PyTorch's native DDP.
"""

from typing import Optional, List, Type, Union, Any
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
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    barrier,
    all_reduce,
    print_rank_0,
)


class DDPTrainer(Trainer):
    """
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

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Initialize scheduler (from base class)
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

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

                # Use no_sync context for gradient accumulation
                # This prevents DDP from synchronizing gradients until the last micro-batch
                context = (
                    self.model.no_sync()
                    if (
                        is_distributed()
                        and num_batches is not None
                        and accumulation_count + 1 < accumulation_target
                    )
                    else torch.enable_grad()
                )

                with context:
                    with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                        logits = self.model(x)
                        loss = self.loss(logits, y)
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
