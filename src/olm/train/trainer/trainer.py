from typing import Type, Optional, Callable, Dict, Any, List, Union

import torch.optim
import torch
from torch.amp import autocast, GradScaler

from olm.data.tokenization import TokenizerBase
from torch.nn import Module
from olm.data.datasets import DataLoader
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.base import LossBase
from olm.train.schedulers.warmup import WarmupCosineScheduler


class TrainerCallback:
    """Base class for trainer callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch_idx: int) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer: "Trainer", batch_idx: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        """Called at the beginning of each optimization step (after gradient accumulation)."""
        pass

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        """Called at the end of each optimization step."""
        pass


class Trainer:
    """
    Manages the training loop for Open Language Model (OLM) architectures.

    This trainer handles the core training logic including:
    - Automatic Mixed Precision (AMP) scaling
    - Gradient accumulation
    - Device management (moving data/models to GPU)
    - Optimization steps
    - Callbacks for validation, checkpointing, and custom logic
    - Learning rate scheduling support
    - Gradient clipping

    Attributes:
        model (Pipeline): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (olm.data.datasets.data_loader.DataLoader): The data provider.
        device (str): The device to train on (e.g., 'cuda', 'cpu').
        context_length (int): The maximum sequence length for training.
        grad_accum_steps (int): Number of steps to accumulate gradients before updating.
        use_amp (bool): Whether to use Automatic Mixed Precision.
        scaler (GradScaler): Gradient scaler for AMP.
        loss (LossBase): The loss function instance.
        callbacks (List[TrainerCallback]): List of callbacks to execute during training.
        scheduler (Optional): Learning rate scheduler to step after each optimization step.
        global_step (int): Current global step count.
        current_epoch (int): Current epoch number.
    """

    def __init__(
        self,
        model: Module,
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
    ):
        """
        Initializes the Trainer.

        Args:
            model (Module): The model architecture to train.
            optimizer (Union[torch.optim.Optimizer, Type[torch.optim.Optimizer]]): The optimizer instance or class.
                If a class is provided, it will be instantiated with `learning_rate` and `weight_decay`.
            dataloader (olm.data.datasets.data_loader.DataLoader): The dataset iterator.
            device (str): Target device ('cuda' or 'cpu').
            context_length (int): Maximum sequence length.
            grad_accum_steps (int, optional): Steps for gradient accumulation. Defaults to 1.
            use_amp (bool, optional): Enable Automatic Mixed Precision. Defaults to True.
            loss (Type[LossBase], optional): Loss function class. Defaults to CrossEntropyLoss.
            callbacks (Optional[List[TrainerCallback]], optional): List of callbacks. Defaults to None.
            scheduler (Optional[Any], optional): Learning rate scheduler. If None and use_warmup_cosine=True,
                a WarmupCosineScheduler will be created. Defaults to None.
            grad_clip_norm (Optional[float], optional): Max norm for gradient clipping. Defaults to None.
            warmup_steps (Optional[int], optional): Number of warmup steps. Defaults to None.
            total_steps (Optional[int], optional): Total training steps. Defaults to None.
            min_lr (float, optional): Minimum learning rate. Defaults to 0.0.
            learning_rate (float, optional): Learning rate for the optimizer (if passing a class). Defaults to 3e-4.
            weight_decay (float, optional): Weight decay coefficient (if passing a class). Defaults to 0.0.
            use_warmup_cosine (bool, optional): Whether to use warmup + cosine scheduling by default. Defaults to True.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.context_length = context_length
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.loss = loss()
        self.losses = []
        self.callbacks = callbacks or []
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.current_epoch = 0
        self.training_state = {
            "current_loss": 0.0,
            "accumulated_loss": 0.0,
        }

        # Initialize Optimizer
        if isinstance(optimizer, type):
            self.optimizer = self._configure_optimizer(
                optimizer, learning_rate, weight_decay
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduling configuration
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.use_warmup_cosine = use_warmup_cosine

        # Set up scheduler (will be finalized in train() if needed)
        self.scheduler = scheduler

    def _configure_optimizer(
        self,
        optimizer_cls: Type[torch.optim.Optimizer],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """
        Configures the optimizer, applying weight decay only to appropriate parameters
        (typically weights of appropriate dimensionality), and excluding biases/LayerNorms.
        """
        # Separate parameters into those that decay and those that don't
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Heuristic: Parameters with 2 or more dimensions (weights) get decay.
            # Parameters with < 2 dimensions (biases, layernorm params) don't.
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = optimizer_cls(optim_groups, lr=learning_rate)
        return optimizer

    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a callback to the trainer."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainerCallback) -> None:
        """Remove a callback from the trainer."""
        self.callbacks.remove(callback)

    def _call_callbacks(self, method_name: str, *args, **kwargs) -> None:
        """Call a specific method on all callbacks."""
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        max_steps: int = None,
        steps_per_epoch: int = None,
    ) -> list[float]:
        """
        Executes the training loop for a specified number of epochs.

        Args:
            epochs (int): The number of complete passes through the dataset.
            log_interval (int): How often to print the loss. Defaults to 10.
            max_steps (int, optional): Maximum number of steps to train for.
            steps_per_epoch (int, optional): Maximum number of steps per epoch. Defaults to None (unlimited).

        Returns:
            list[float]: A list of recorded loss values.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Initialize default warmup + cosine scheduler if not provided
        if self.scheduler is None and self.use_warmup_cosine:
            # Calculate total steps if not provided
            if self.total_steps is None:
                if max_steps is not None:
                    self.total_steps = max_steps
                elif steps_per_epoch is not None:
                    self.total_steps = epochs * steps_per_epoch
                else:
                    # Estimate based on dataloader size (if available)
                    try:
                        dataset_size = len(self.dataloader)
                        self.total_steps = epochs * dataset_size
                    except:
                        # Default to a reasonable number for streaming datasets
                        self.total_steps = epochs * 10000

            # Calculate warmup steps if not provided (10% of total steps, min 100, max 5000)
            if self.warmup_steps is None:
                self.warmup_steps = min(max(int(0.1 * self.total_steps), 100), 5000)

            # Create warmup + cosine scheduler
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
                min_lr=self.min_lr,
            )
            print(
                f"Initialized WarmupCosineScheduler: warmup_steps={self.warmup_steps}, total_steps={self.total_steps}, min_lr={self.min_lr}",
                flush=True,
            )

        losses = []

        # Call on_train_begin callbacks
        self._call_callbacks("on_train_begin", self)

        print(f"{'Epoch':^6} | {'Step':^8} | {'Loss':^10}", flush=True)
        print("-" * 30, flush=True)

        for epoch in range(epochs):
            self.current_epoch = epoch
            self._call_callbacks("on_epoch_begin", self, epoch)

            accumulated_loss = 0.0
            epoch_step = 0  # Track steps within this epoch

            for step, (x, y) in enumerate(self.dataloader):
                self._call_callbacks("on_batch_begin", self, step)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)  # (B, T, V)
                    loss = self.loss(logits, y)
                    loss_val = loss.item()
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                accumulated_loss += loss_val

                self.training_state["current_loss"] = loss_val
                self.training_state["accumulated_loss"] = accumulated_loss

                self._call_callbacks("on_batch_end", self, step, loss_val)

                if (step + 1) % self.grad_accum_steps == 0:
                    self._call_callbacks("on_step_begin", self, self.global_step)

                    # Gradient clipping
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Learning rate scheduling
                    if self.scheduler is not None:
                        self.scheduler.step()

                    avg_loss = accumulated_loss / self.grad_accum_steps
                    self.global_step += 1

                    if self.global_step % log_interval == 0:
                        losses.append(avg_loss)
                        self.losses.append(avg_loss)
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        print(
                            f"{epoch+1:^6} | {self.global_step:^8} | {avg_loss:^10.4f} | LR: {current_lr:.2e}",
                            flush=True,
                        )

                    self._call_callbacks(
                        "on_step_end", self, self.global_step, avg_loss
                    )

                    # Reset accumulated loss
                    accumulated_loss = 0.0
                    epoch_step += 1

                    # Check if we've reached max_steps
                    if max_steps and self.global_step >= max_steps:
                        self._call_callbacks("on_epoch_end", self, epoch)
                        self._call_callbacks("on_train_end", self)
                        print("-" * 30, flush=True)
                        return losses

                    # Check if we've reached steps_per_epoch
                    if steps_per_epoch and epoch_step >= steps_per_epoch:
                        break

            self._call_callbacks("on_epoch_end", self, epoch)

        self._call_callbacks("on_train_end", self)
        print("-" * 30, flush=True)
        self.losses = losses
        return losses
