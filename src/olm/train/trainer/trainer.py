from typing import Type, Optional, Callable, Dict, Any, List, Tuple, Union
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import os
import torch.optim
import torch
import time
import math
import warnings
from torch.amp import autocast, GradScaler

from olm.data.tokenization import TokenizerBase
from torch.nn import Module
from olm.data.datasets import DataLoader
from olm.core.dist import get_rank, get_world_size, is_main_process
from olm.core.seed import get_rng_state, set_rng_state
from olm.train.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    add_legacy_aliases,
    find_latest_checkpoint,
    read_checkpoint,
    unwrap_model,
    write_checkpoint,
)
from olm.train.losses.cross_entropy import CrossEntropyLoss
from olm.train.losses.base import LossBase
from olm.train.schedulers.warmup import WarmupCosineScheduler
from olm.train.output import LMOutput, as_lm_output


class TrainerCallback:
    """
    Base class for trainer callbacks.

    Callbacks run in ascending ``order``; callbacks with the same ``order`` run
    in the order they were passed. The built-in ``ThroughputCallback`` (-20)
    and ``ValidationCallback`` (-10) run first, so every other callback sees
    this step's timing and validation result, and ``CheckpointCallback`` (10)
    runs last, so a checkpoint holds the state every other callback reached in
    the same step.
    """

    order: int = 0

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

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state to store in training checkpoints.

        The default stores nothing. Override it together with
        :meth:`load_state_dict` when the callback's decisions depend on earlier
        steps, such as the best validation loss seen so far. Use only tensors
        and plain Python values, so the checkpoint loads with
        ``torch.load(weights_only=True)``.
        """
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the state returned by :meth:`state_dict`."""
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
    - Saving and resuming complete training state

    Training contract:
        The dataloader must yield ``(input_ids, labels)``. Both tensors are
        moved to ``device``. The model must return logits shaped
        ``[batch, seq_len, vocab_size]`` (or an ``LMOutput``/dict holding them)
        so the configured loss can compare logits against ``labels`` shaped
        ``[batch, seq_len]``.

    Resuming:
        ``save_checkpoint(path)`` writes the model, optimizer, scheduler,
        GradScaler, step counters, random number generator states, data
        position and callback state. ``Trainer(..., resume_from=path)`` or
        ``load_checkpoint(path)`` restores them, and the next ``train()`` call
        with the same arguments continues the interrupted run: it starts in
        the saved epoch and skips the batches that epoch already consumed.

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
        global_step (int): Number of optimizer steps taken.
        current_epoch (int): Current epoch number.
        epoch_step (int): Optimizer steps taken in the current epoch.
        batches_in_epoch (int): Batches of the current epoch used by completed
            optimizer steps; a resumed run skips this many.
        total_tokens_processed (int): Total number of tokens processed during training.
        losses (List[float]): Loss of every optimizer step, including the steps
            before a resume.
        step_start_time (float): Timestamp of the current step start.
        training_start_time (float): Timestamp when training began.
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
        resume_from: Optional[Union[str, os.PathLike]] = None,
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
            total_steps (Optional[int], optional): Total optimizer steps of the default schedule.
                Defaults to None (derived from the arguments of `train()`).
            min_lr (float, optional): Minimum learning rate. Defaults to 0.0.
            learning_rate (float, optional): Learning rate for the optimizer (if passing a class). Defaults to 3e-4.
            weight_decay (float, optional): Weight decay coefficient (if passing a class). Defaults to 0.0.
            use_warmup_cosine (bool, optional): Whether to use warmup + cosine scheduling by default. Defaults to True.
            resume_from (Optional[Union[str, os.PathLike]], optional): Checkpoint to resume from. A file
                path loads that checkpoint. A directory loads its newest `step_<N>.pt` file, and
                `"latest"` does the same for the directory of the first `CheckpointCallback`;
                when no checkpoint exists yet, both start a new run. Defaults to None.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.context_length = context_length
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = GradScaler(
            self.device_type, enabled=use_amp and self.device_type == "cuda"
        )
        self.loss = loss() if isinstance(loss, type) else loss
        self.mtp_loss = mtp_loss() if isinstance(mtp_loss, type) else mtp_loss
        self.losses = []
        self.callbacks = callbacks or []
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.current_epoch = 0
        self.epoch_step = 0
        self.batches_in_epoch = 0
        self.total_tokens_processed = 0
        self.step_start_time = None
        self.training_start_time = None
        self.training_state = {
            "current_loss": 0.0,
            "accumulated_loss": 0.0,
            "perplexity": 0.0,
            "tokens_per_sec": 0.0,
            "learning_rate": 0.0,
            "total_tokens": 0,
        }

        # Data position bookkeeping for checkpoints (see _start_epoch).
        self._epoch_complete = False
        self._epoch_start_rng = None
        self._data_iterator_reused = False
        self._last_data_iterator = None
        self._pending_resume = None

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

        if resume_from is not None:
            checkpoint_path = self._resolve_resume_path(resume_from)
            if checkpoint_path is not None:
                self.load_checkpoint(checkpoint_path)

    def _sum_aux_losses(
        self,
        aux_losses: Optional[
            torch.Tensor | List[torch.Tensor] | Dict[str, torch.Tensor]
        ],
        device: torch.device,
    ) -> torch.Tensor:
        if aux_losses is None:
            return torch.zeros((), device=device)

        if torch.is_tensor(aux_losses):
            return aux_losses.to(device)

        if isinstance(aux_losses, dict):
            losses = list(aux_losses.values())
        else:
            losses = list(aux_losses)

        if not losses:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        for loss in losses:
            total = total + loss.to(device)
        return total

    def _compute_model_loss(
        self,
        model_output: torch.Tensor | LMOutput | Dict[str, Any],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, LMOutput]:
        output = as_lm_output(model_output)
        loss = self.loss(output.logits, labels)

        if output.mtp_logits is not None and self.mtp_loss is not None:
            loss = loss + self.mtp_loss(output.mtp_logits, labels)

        loss = loss + self._sum_aux_losses(output.aux_losses, loss.device)
        return loss, output

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
        """Call a specific method on all callbacks, in ascending ``order``."""
        for callback in sorted(self.callbacks, key=lambda cb: getattr(cb, "order", 0)):
            getattr(callback, method_name)(*args, **kwargs)

    def _should_stop_training(self) -> bool:
        """Return True when a callback has requested early termination."""
        return bool(self.training_state.get("should_stop", False))

    # --- Hooks that DDPTrainer and FSDPTrainer override ---

    def _prepare_dataloader_for_epoch(self, epoch: int) -> None:
        """Set per-epoch dataloader state before the epoch is iterated."""
        pass

    def _accumulation_context(
        self, sync_gradients: bool, num_batches: Optional[int]
    ) -> AbstractContextManager:
        """Context for one micro-batch's forward and backward pass."""
        return nullcontext()

    def _clip_gradients(self) -> None:
        """Clip the gradient norm to ``grad_clip_norm``."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

    def _reduce_step_metrics(
        self, avg_loss: float, tokens_per_sec: float
    ) -> Tuple[float, float]:
        """Combine the per-process loss and throughput of one step."""
        return avg_loss, tokens_per_sec

    def _gather_rng_states(self) -> List[Dict[str, Any]]:
        """Return the RNG states of every process, indexed by rank."""
        return [self._rng_state()]

    # --- Learning rate schedule ---

    def _default_total_steps(
        self,
        epochs: int,
        max_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> int:
        """
        Number of optimizer steps ``train(epochs, max_steps, steps_per_epoch)`` will take.

        The scheduler steps once per optimizer step, so with gradient
        accumulation an epoch of ``N`` batches is ``ceil(N / grad_accum_steps)``
        steps, not ``N``.
        """
        try:
            num_batches = len(self.dataloader)
        except TypeError:
            num_batches = None

        if num_batches is not None:
            steps_in_epoch = math.ceil(num_batches / self.grad_accum_steps)
            if steps_per_epoch is not None:
                steps_in_epoch = min(steps_in_epoch, steps_per_epoch)
            total_steps = epochs * steps_in_epoch
            if max_steps is not None:
                total_steps = min(total_steps, max_steps)
            return total_steps

        if max_steps is not None:
            return max_steps
        if steps_per_epoch is not None:
            return epochs * steps_per_epoch

        warnings.warn(
            "The dataloader has no length, so the default warmup + cosine schedule "
            "assumes 10000 optimizer steps per epoch. Pass max_steps or "
            "steps_per_epoch to train(), or total_steps to the Trainer, to set "
            "the schedule length.",
            stacklevel=3,
        )
        return epochs * 10000

    def _initialize_scheduler(
        self,
        epochs: int,
        max_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        """Initialize the default warmup + cosine scheduler."""
        if self.total_steps is None:
            self.total_steps = self._default_total_steps(
                epochs, max_steps, steps_per_epoch
            )

        if self.warmup_steps is None:
            self.warmup_steps = min(
                max(int(0.1 * self.total_steps), 1),
                5000,
                self.total_steps,
            )

        self._build_default_scheduler()

    def _build_default_scheduler(self) -> None:
        """Create the WarmupCosineScheduler from ``warmup_steps`` and ``total_steps``."""
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr=self.min_lr,
        )
        if is_main_process():
            print(
                f"Initialized WarmupCosineScheduler: warmup_steps={self.warmup_steps}, "
                f"total_steps={self.total_steps}, min_lr={self.min_lr}",
                flush=True,
            )

    # --- Data position ---

    def _data_generators(self) -> List[torch.Generator]:
        """The dataloader's own random generators (shuffling and worker seeds)."""
        generators = []
        for owner in (self.dataloader, getattr(self.dataloader, "sampler", None)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and not any(
                generator is known for known in generators
            ):
                generators.append(generator)
        return generators

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Global RNG states plus the dataloader's own generators."""
        state = get_rng_state()
        state["data_generators"] = [g.get_state() for g in self._data_generators()]
        return state

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        """Restore a state captured by :meth:`_capture_rng_state`."""
        set_rng_state(state)
        saved = state.get("data_generators") or []
        generators = self._data_generators()
        if len(saved) == len(generators):
            for generator, generator_state in zip(generators, saved):
                generator.set_state(generator_state)
        elif saved:
            warnings.warn(
                "The checkpoint holds the state of a different number of dataloader "
                "generators than this dataloader has, so its shuffle order is not "
                "restored.",
                stacklevel=3,
            )

    def _start_epoch(self, resume: Optional[Dict[str, Any]]) -> Any:
        """
        Create the epoch's data iterator, fast-forwarded when resuming.

        A shuffled map-style dataloader draws its permutation, and a
        multi-worker one its worker seeds, from random generators when the
        iterator is created. Restoring the generators to their state at the
        start of the interrupted epoch before creating the iterator reproduces
        its batch order; the batches it had already used are then skipped and
        the generators are set to their state at save time.
        """
        loader = self.dataloader
        persistent = (
            getattr(loader, "persistent_workers", False)
            and getattr(loader, "num_workers", 0) > 0
        )

        if resume is not None:
            # With persistent workers, only the first iter() in a process
            # creates an iterator (and draws a worker seed); later epochs reset
            # it. Make the iter() below do what it did in the interrupted run.
            existing = getattr(loader, "_iterator", self._last_data_iterator)
            if persistent and resume["data_iterator_reused"] and existing is None:
                # The interrupted run reset an iterator in this epoch: create
                # one now, so the iter() below is a reset too.
                self._last_data_iterator = iter(loader)
            elif (
                persistent
                and not resume["data_iterator_reused"]
                and existing is not None
                and hasattr(loader, "_iterator")
            ):
                # The interrupted run created this epoch's iterator, but the
                # loader still holds one from an earlier train() call in this
                # process (load_checkpoint on a trainer that already trained):
                # shut it down, so the iter() below creates a new one.
                if hasattr(existing, "_shutdown_workers"):
                    existing._shutdown_workers()
                loader._iterator = None
                self._last_data_iterator = None
            if resume["epoch_start_rng"] is not None:
                self._restore_rng_state(resume["epoch_start_rng"])
            elif resume["current_rng"] is not None:
                self._restore_rng_state(resume["current_rng"])

        self._epoch_start_rng = self._capture_rng_state()
        data_iter = iter(loader)
        self._data_iterator_reused = (
            persistent and data_iter is self._last_data_iterator
        )
        self._last_data_iterator = data_iter if persistent else None

        if resume is not None:
            skip = self.batches_in_epoch
            if skip > 0:
                if is_main_process():
                    print(
                        f"Skipping {skip} batches to resume epoch "
                        f"{self.current_epoch + 1} where the checkpoint left off",
                        flush=True,
                    )
                for _ in range(skip):
                    if next(data_iter, None) is None:
                        break
            if (
                resume["epoch_start_rng"] is not None
                and resume["current_rng"] is not None
            ):
                self._restore_rng_state(resume["current_rng"])

        return data_iter

    # --- Training loop ---

    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        max_steps: int = None,
        steps_per_epoch: int = None,
    ) -> list[float]:
        """
        Executes the training loop for a specified number of epochs.

        After a resume, call it with the same arguments as the interrupted
        run: ``epochs`` and ``max_steps`` count from the start of that run, so
        training continues from the saved epoch and step.

        Args:
            epochs (int): The number of complete passes through the dataset.
            log_interval (int): How often to print the loss. Defaults to 10.
            max_steps (int, optional): Maximum number of optimizer steps to train for.
            steps_per_epoch (int, optional): Maximum number of optimizer steps per epoch.
                Defaults to None (unlimited).

        Returns:
            list[float]: The loss of each optimizer step taken by this call.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is None and self.use_warmup_cosine:
            self._initialize_scheduler(epochs, max_steps, steps_per_epoch)

        losses = []
        resume = self._pending_resume
        self._pending_resume = None
        start_epoch = self.current_epoch if resume is not None else 0

        def finish_accumulation(actual_count: int, epoch_ended: bool) -> bool:
            nonlocal accumulated_loss, accumulated_tokens, accumulation_count

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
                self._clip_gradients()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = accumulated_loss / actual_count
            self.global_step += 1
            self.epoch_step += 1
            self.batches_in_epoch += actual_count
            self._epoch_complete = (
                epoch_ended
                or (num_batches is not None and self.batches_in_epoch >= num_batches)
                or bool(steps_per_epoch and self.epoch_step >= steps_per_epoch)
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            step_time = (
                time.time() - self.step_start_time if self.step_start_time else 0.0
            )
            tokens_per_sec = accumulated_tokens / step_time if step_time > 0 else 0.0
            avg_loss, tokens_per_sec = self._reduce_step_metrics(
                avg_loss, tokens_per_sec
            )
            perplexity = math.exp(min(avg_loss, 20))

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
                    f"{epoch+1:^6} | {self.global_step:^8} | {avg_loss:^10.4f} | {perplexity:^11.2f} | {tokens_per_sec:^10.0f} | {current_lr:^10.2e}",
                    flush=True,
                )

            self._call_callbacks("on_step_end", self, self.global_step, avg_loss)
            should_stop = self._should_stop_training() or bool(
                max_steps and self.global_step >= max_steps
            )

            accumulated_loss = 0.0
            accumulated_tokens = 0
            accumulation_count = 0

            return should_stop

        # Call on_train_begin callbacks
        self._call_callbacks("on_train_begin", self)

        # Initialize training start time
        self.training_start_time = time.time()

        if is_main_process():
            print(
                f"{'Epoch':^6} | {'Step':^8} | {'Loss':^10} | {'Perplexity':^11} | {'Tokens/s':^10} | {'LR':^10}",
                flush=True,
            )
            print("-" * 80, flush=True)

        stop = self._should_stop_training() or bool(
            max_steps and self.global_step >= max_steps
        )

        for epoch in range(start_epoch, epochs):
            if stop:
                break

            self.current_epoch = epoch
            if resume is None:
                self.epoch_step = 0
                self.batches_in_epoch = 0
                self._epoch_complete = False
            self._prepare_dataloader_for_epoch(epoch)
            self._call_callbacks("on_epoch_begin", self, epoch)

            accumulated_loss = 0.0
            accumulated_tokens = 0
            accumulation_count = 0
            accumulation_target = self.grad_accum_steps
            try:
                num_batches = len(self.dataloader)
            except TypeError:
                num_batches = None

            data_iter = self._start_epoch(resume)
            first_batch = self.batches_in_epoch
            resume = None

            if steps_per_epoch and self.epoch_step >= steps_per_epoch:
                data_iter = iter(())

            for step, (x, y) in enumerate(data_iter, start=first_batch):
                self._call_callbacks("on_batch_begin", self, step)

                # Start timing for this step if it's the first batch in accumulation
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

                # Track tokens processed
                batch_tokens = x.numel()
                self.total_tokens_processed += batch_tokens
                accumulated_tokens += batch_tokens

                sync_gradients = accumulation_count + 1 >= accumulation_target
                with self._accumulation_context(sync_gradients, num_batches):
                    with autocast(self.device_type, enabled=self.use_amp):
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

                if accumulation_count == accumulation_target:
                    stop = finish_accumulation(accumulation_count, epoch_ended=False)
                    if stop or self._epoch_complete:
                        break

            if not stop and accumulation_count > 0:
                stop = finish_accumulation(accumulation_count, epoch_ended=True)
            if not stop:
                self._epoch_complete = True

            self._call_callbacks("on_epoch_end", self, epoch)

        self._call_callbacks("on_train_end", self)
        if is_main_process():
            self._print_training_summary()
        return losses

    # --- Checkpoints ---

    def _rng_state(self) -> Dict[str, Any]:
        """This process's RNG states now and at the start of the current epoch."""
        return {
            "current": self._capture_rng_state(),
            "epoch_start": self._epoch_start_rng,
        }

    def _callback_keys(self) -> List[str]:
        """Checkpoint keys of the callbacks: the class name, numbered when repeated."""
        counts = {}
        keys = []
        for callback in self.callbacks:
            name = type(callback).__name__
            index = counts.get(name, 0)
            counts[name] = index + 1
            keys.append(name if index == 0 else f"{name}_{index}")
        return keys

    def _build_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        rng_states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble a format version 1 checkpoint (see ``olm.train.checkpointing``)."""
        from olm import __version__ as olm_version

        callback_states = {}
        for key, callback in zip(self._callback_keys(), self.callbacks):
            state_dict = getattr(callback, "state_dict", None)
            state = state_dict() if callable(state_dict) else {}
            if state:
                callback_states[key] = state

        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "olm_version": olm_version,
            "torch_version": str(torch.__version__),
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scheduler_class": (
                type(self.scheduler).__name__ if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict(),
            "trainer": {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "epoch_step": self.epoch_step,
                "batches_in_epoch": self.batches_in_epoch,
                "epoch_complete": self._epoch_complete,
                "data_iterator_reused": self._data_iterator_reused,
                "total_tokens_processed": self.total_tokens_processed,
                "grad_accum_steps": self.grad_accum_steps,
                "world_size": get_world_size(),
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "should_stop": bool(self.training_state.get("should_stop", False)),
            },
            "rng": rng_states,
            "callbacks": callback_states,
            "losses": list(self.losses),
        }
        return add_legacy_aliases(checkpoint)

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the complete training state as a format version 1 checkpoint dict.

        It holds only this process's RNG state. Use :meth:`save_checkpoint` to
        write a file; under DDP it also collects the RNG state of every rank.

        Returns:
            The checkpoint dict (keys are described in ``olm.train.checkpointing``).
        """
        return self._build_checkpoint(
            unwrap_model(self.model).state_dict(),
            self.optimizer.state_dict(),
            [self._rng_state()],
        )

    def save_checkpoint(self, path: Union[str, os.PathLike]) -> Optional[Path]:
        """
        Write the complete training state to ``path``.

        The file is written atomically (a temporary file renamed over
        ``path``). Under DDP every rank must call this; rank 0 writes the file.

        Args:
            path: Destination file, conventionally ending in ``.pt``.

        Returns:
            ``path`` as a ``Path`` on the process that wrote it, else ``None``.
        """
        rng_states = self._gather_rng_states()
        if not is_main_process():
            return None
        checkpoint = self._build_checkpoint(
            unwrap_model(self.model).state_dict(),
            self.optimizer.state_dict(),
            rng_states,
        )
        return write_checkpoint(checkpoint, path)

    def load_checkpoint(
        self,
        path: Union[str, os.PathLike],
        strict: bool = True,
        weights_only: bool = True,
    ) -> None:
        """
        Restore the training state saved by :meth:`save_checkpoint`.

        Files written by ``CheckpointCallback`` before format version 1 load
        too; they hold no RNG state or data position, so a resumed run is
        close to the interrupted one but not identical (see the
        checkpointing guide).

        Args:
            path: Checkpoint file.
            strict: Passed to the model's ``load_state_dict``.
            weights_only: Passed to ``torch.load``; see
                ``olm.train.checkpointing.read_checkpoint``.
        """
        checkpoint = read_checkpoint(
            path, map_location="cpu", weights_only=weights_only
        )
        self.load_state_dict(checkpoint, strict=strict)
        if is_main_process():
            print(
                f"Resumed from {path} after step {self.global_step}; training "
                f"continues in epoch {self.current_epoch + 1}",
                flush=True,
            )

    def _load_model_state_dict(self, state: Dict[str, Any], strict: bool) -> None:
        unwrap_model(self.model).load_state_dict(state, strict=strict)

    def _load_optimizer_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    def load_state_dict(self, checkpoint: Dict[str, Any], strict: bool = True) -> None:
        """
        Restore the training state from a checkpoint dict.

        Args:
            checkpoint: A dict from :meth:`state_dict` or
                ``olm.train.checkpointing.read_checkpoint``.
            strict: Passed to the model's ``load_state_dict``.
        """
        trainer_state = checkpoint["trainer"]
        scheduler_state = checkpoint.get("scheduler")

        self._load_model_state_dict(checkpoint["model"], strict)

        # Build the default scheduler before restoring the optimizer:
        # constructing a scheduler overwrites every parameter group's
        # learning rate, which the optimizer state then restores.
        if scheduler_state is not None and self.scheduler is None:
            self._restore_default_scheduler(checkpoint)

        if checkpoint.get("optimizer") is not None:
            self._load_optimizer_state_dict(checkpoint["optimizer"])
        else:
            warnings.warn(
                "The checkpoint holds no optimizer state, so the optimizer starts "
                "fresh.",
                stacklevel=2,
            )

        if scheduler_state is not None and self.scheduler is not None:
            saved_class = checkpoint.get("scheduler_class")
            if saved_class is not None and saved_class != type(self.scheduler).__name__:
                warnings.warn(
                    f"The checkpoint's scheduler state comes from {saved_class}, but "
                    f"this Trainer uses {type(self.scheduler).__name__}.",
                    stacklevel=2,
                )
            self.scheduler.load_state_dict(scheduler_state)
        elif (
            scheduler_state is None
            and self.use_warmup_cosine
            and self.scheduler is None
        ):
            warnings.warn(
                "The checkpoint holds no scheduler state, so train() will start a "
                "new warmup + cosine schedule from step 0.",
                stacklevel=2,
            )

        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = trainer_state["global_step"]
        self.current_epoch = trainer_state["epoch"]
        self.epoch_step = trainer_state.get("epoch_step", 0)
        self.batches_in_epoch = trainer_state.get("batches_in_epoch", 0)
        self.total_tokens_processed = trainer_state.get("total_tokens_processed", 0)
        self.training_state["total_tokens"] = self.total_tokens_processed
        self.training_state["should_stop"] = trainer_state.get("should_stop", False)
        self._epoch_complete = trainer_state.get("epoch_complete", False)

        if checkpoint["format_version"] == 0:
            self._estimate_legacy_data_position()

        saved_accum = trainer_state.get("grad_accum_steps", self.grad_accum_steps)
        saved_world_size = trainer_state.get("world_size", get_world_size())
        if saved_accum != self.grad_accum_steps or saved_world_size != get_world_size():
            warnings.warn(
                f"The checkpoint was written with grad_accum_steps={saved_accum} and "
                f"world_size={saved_world_size}, but this run uses "
                f"grad_accum_steps={self.grad_accum_steps} and "
                f"world_size={get_world_size()}. Training continues, but the data "
                "position is only exact when both match.",
                stacklevel=2,
            )

        rng = self._rank_rng_state(checkpoint.get("rng"))
        if rng is not None:
            self._restore_rng_state(rng["current"])
        self._pending_resume = {
            "current_rng": rng["current"] if rng is not None else None,
            "epoch_start_rng": rng["epoch_start"] if rng is not None else None,
            "data_iterator_reused": trainer_state.get("data_iterator_reused", False),
        }

        if self._epoch_complete:
            # The saved epoch used all its data: continue with the next one,
            # whose iterator the interrupted run created from the RNG state at
            # save time (and, with persistent workers, by resetting the
            # existing iterator).
            self.current_epoch += 1
            self.epoch_step = 0
            self.batches_in_epoch = 0
            self._epoch_complete = False
            self._pending_resume["epoch_start_rng"] = None
            self._pending_resume["data_iterator_reused"] = True

        saved_callbacks = checkpoint.get("callbacks") or {}
        for key, callback in zip(self._callback_keys(), self.callbacks):
            if key in saved_callbacks and hasattr(callback, "load_state_dict"):
                callback.load_state_dict(saved_callbacks[key])

        # Only the main process records losses during training.
        self.losses = list(checkpoint.get("losses") or []) if is_main_process() else []

    def _restore_default_scheduler(self, checkpoint: Dict[str, Any]) -> None:
        """Rebuild the default WarmupCosineScheduler of the run that wrote ``checkpoint``."""
        trainer_state = checkpoint["trainer"]
        scheduler_state = checkpoint["scheduler"]
        saved_class = checkpoint.get("scheduler_class")
        is_default = saved_class == "WarmupCosineScheduler" or (
            saved_class is None
            and "total_steps" in scheduler_state
            and "warmup_steps" in scheduler_state
        )
        if not self.use_warmup_cosine or not is_default:
            warnings.warn(
                f"The checkpoint holds the state of a {saved_class or 'custom'} "
                "scheduler, but this Trainer has no scheduler to load it into. Pass "
                "the scheduler you trained with as scheduler=... to restore it.",
                stacklevel=3,
            )
            return

        self.total_steps = trainer_state.get("total_steps") or scheduler_state.get(
            "total_steps"
        )
        self.warmup_steps = trainer_state.get("warmup_steps") or scheduler_state.get(
            "warmup_steps"
        )
        self.min_lr = scheduler_state.get("min_lr", self.min_lr)
        self._build_default_scheduler()

    def _estimate_legacy_data_position(self) -> None:
        """
        Estimate the data position of a checkpoint written before format version 1.

        Those files record only the step and epoch. In the first epoch every
        optimizer step used ``grad_accum_steps`` batches, so the position is
        known; later epochs restart from their first batch.
        """
        if self.current_epoch == 0:
            self.epoch_step = self.global_step
            self.batches_in_epoch = self.global_step * self.grad_accum_steps
            message = (
                f"it resumes after {self.batches_in_epoch} batches, assuming "
                "gradient accumulation and batch size did not change"
            )
        else:
            message = f"epoch {self.current_epoch + 1} restarts from its first batch"
        warnings.warn(
            "This checkpoint predates checkpoint format version 1 and records no "
            f"data position or RNG state: {message}.",
            stacklevel=3,
        )

    def _rank_rng_state(
        self, rng_states: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Pick this rank's entry from the checkpoint's per-rank RNG states."""
        if not rng_states:
            return None
        rank = get_rank()
        if len(rng_states) != get_world_size():
            warnings.warn(
                f"The checkpoint holds RNG states for {len(rng_states)} process(es) "
                f"but this run has {get_world_size()}; processes without a saved "
                "state use rank 0's.",
                stacklevel=3,
            )
        return rng_states[rank] if rank < len(rng_states) else rng_states[0]

    def _resolve_resume_path(
        self, resume_from: Union[str, os.PathLike]
    ) -> Optional[Path]:
        """Turn ``resume_from`` into a checkpoint file, or ``None`` to start fresh."""
        if str(resume_from) == "latest":
            directories = [
                cb.checkpoint_dir
                for cb in self.callbacks
                if hasattr(cb, "checkpoint_dir")
            ]
            if not directories:
                raise ValueError(
                    'resume_from="latest" looks for checkpoints in the directory of '
                    "a CheckpointCallback, but this Trainer has none. Pass a "
                    "checkpoint file or directory instead."
                )
            directory = Path(directories[0])
        elif Path(resume_from).is_dir():
            directory = Path(resume_from)
        elif Path(resume_from).is_file():
            return Path(resume_from)
        else:
            raise FileNotFoundError(f"No checkpoint file or directory at {resume_from}")

        latest = find_latest_checkpoint(directory)
        if latest is None and is_main_process():
            print(
                f"No checkpoint found in {directory}; starting a new run.", flush=True
            )
        return latest

    def _print_training_summary(self) -> None:
        """Print a summary of training metrics."""
        print("-" * 80, flush=True)
        print("\nTraining Summary:", flush=True)
        print(f"  Total Steps: {self.global_step}", flush=True)
        print(f"  Total Tokens Processed: {self.total_tokens_processed:,}", flush=True)

        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"  Total Training Time: {hours}h {minutes}m {seconds}s", flush=True)

            avg_tokens_per_sec = self.total_tokens_processed / total_time
            print(
                f"  Average Throughput: {avg_tokens_per_sec:.0f} tokens/sec", flush=True
            )

        if self.losses:
            final_loss = self.losses[-1]
            final_perplexity = math.exp(min(final_loss, 20))
            print(f"  Final Loss: {final_loss:.4f}", flush=True)
            print(f"  Final Perplexity: {final_perplexity:.2f}", flush=True)

        print("-" * 80, flush=True)
