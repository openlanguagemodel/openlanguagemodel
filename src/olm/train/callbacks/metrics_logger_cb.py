"""
Metrics logging callback for tracking training metrics.
"""

import json
from pathlib import Path
from olm.train.trainer import TrainerCallback


class MetricsLoggerCallback(TrainerCallback):
    """
    Callback to log metrics to a JSONL file.

    Args:
        log_dir: Directory to save logs.
        log_every: Log metrics every N steps.
    """

    def __init__(self, log_dir: str = "logs", log_every: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_every = log_every

        # Create metrics file
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Log metrics after each optimization step if needed."""
        if step % self.log_every == 0:
            metrics = {
                "step": step,
                "epoch": trainer.current_epoch,
                "train_loss": loss,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            }

            # Add validation loss if available
            if "val_loss" in trainer.training_state:
                metrics["val_loss"] = trainer.training_state["val_loss"]

            # Write to file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
