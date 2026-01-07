"""
Throughput monitoring callback.
"""

import time
from olm.train.trainer import TrainerCallback


class ThroughputCallback(TrainerCallback):
    """
    Callback to monitor training throughput (tokens/sec, samples/sec).

    Args:
        log_every: Log throughput every N steps.
        context_length: Length of each sequence.
        batch_size: Total batch size (including gradient accumulation).
    """

    def __init__(
        self,
        log_every: int = 100,
        context_length: int = 1024,
        batch_size: int = 8,
    ):
        self.log_every = log_every
        self.context_length = context_length
        self.batch_size = batch_size
        self.step_start_time = None

    def on_step_begin(self, trainer, step: int) -> None:
        """Record start time of the step."""
        self.step_start_time = time.time()

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Calculate and log throughput."""
        if step % self.log_every == 0 and self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            tokens_per_sec = (self.batch_size * self.context_length) / step_time
            samples_per_sec = self.batch_size / step_time

            print(
                f"[Throughput] Step {step}: "
                f"{tokens_per_sec:.0f} tokens/s, "
                f"{samples_per_sec:.2f} samples/s"
            )

            # Store in trainer state
            trainer.training_state["tokens_per_sec"] = tokens_per_sec
            trainer.training_state["samples_per_sec"] = samples_per_sec
