"""
Throughput monitoring callback.
"""

import time
import warnings
from typing import Optional

import torch

from olm.core.dist import all_reduce, is_distributed, is_main_process
from olm.train.trainer import TrainerCallback


class ThroughputCallback(TrainerCallback):
    """
    Callback to monitor training throughput (tokens/sec, samples/sec).

    A step is timed from the moment its first micro-batch starts to the end of
    the optimizer step, so the forward and backward passes of every
    micro-batch are included. Tokens are counted from the batches the Trainer
    actually processed (summed over ranks under DDP and FSDP). The callback
    runs before the other callbacks (``order = -20``), so validation and
    checkpoint saving in the same step do not count as training time.

    Args:
        log_every: Log throughput every N steps.
        context_length: Tokens per sample, used to turn tokens/s into
            samples/s. Defaults to None (``trainer.context_length``).
        batch_size: Deprecated and ignored: tokens are counted from the
            batches.
    """

    order = -20

    def __init__(
        self,
        log_every: int = 100,
        context_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if batch_size is not None:
            warnings.warn(
                "ThroughputCallback(batch_size=...) is deprecated and ignored: "
                "tokens are now counted from the batches the Trainer processes.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.log_every = log_every
        self.context_length = context_length
        self.batch_size = batch_size
        self.step_start_time = None
        self.step_start_tokens = 0

    def on_batch_begin(self, trainer, batch_idx: int) -> None:
        """Record the start of a step at its first micro-batch."""
        if self.step_start_time is None:
            self.step_start_time = time.perf_counter()
            self.step_start_tokens = trainer.total_tokens_processed

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Calculate and log throughput."""
        if self.step_start_time is None:
            return
        step_time = time.perf_counter() - self.step_start_time
        tokens = trainer.total_tokens_processed - self.step_start_tokens
        self.step_start_time = None

        if step % self.log_every != 0:
            return

        if is_distributed():
            token_count = torch.tensor([float(tokens)], device=trainer.device)
            all_reduce(token_count)
            tokens = token_count.item()

        tokens_per_sec = tokens / step_time if step_time > 0 else 0.0
        samples_per_sec = tokens_per_sec / (
            self.context_length or trainer.context_length
        )

        if is_main_process():
            print(
                f"[Throughput] Step {step}: "
                f"{tokens_per_sec:.0f} tokens/s, "
                f"{samples_per_sec:.2f} samples/s"
            )

        # Store in trainer state
        trainer.training_state["tokens_per_sec"] = tokens_per_sec
        trainer.training_state["samples_per_sec"] = samples_per_sec
