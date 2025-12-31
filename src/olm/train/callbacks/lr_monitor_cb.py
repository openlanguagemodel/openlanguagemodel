"""
Learning rate monitoring callback.
"""

from olm.train.trainer import TrainerCallback


class LRMonitorCallback(TrainerCallback):
    """
    Callback to monitor and log learning rate.

    Args:
        log_every: Log learning rate every N steps.
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Log learning rate after each optimization step if needed."""
        if step % self.log_every == 0:
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"[LR Monitor] Step {step}: Learning rate = {current_lr:.2e}")
