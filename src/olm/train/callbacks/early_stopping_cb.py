"""
Early stopping callback to prevent overfitting.
"""

from olm.train.trainer import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback to stop training early if validation loss doesn't improve.

    Args:
        patience: Number of validation checks to wait for improvement.
        min_delta: Minimum change in validation loss to qualify as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_step = 0

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Check for early stopping after each step."""
        if "val_loss" in trainer.training_state:
            val_loss = trainer.training_state["val_loss"]

            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1

                if self.wait >= self.patience:
                    self.stopped_step = step
                    print(
                        f"[Early Stopping] Stopping at step {step} due to no improvement"
                    )
                    # Signal to stop training (trainer needs to handle this)
                    trainer.training_state["should_stop"] = True
