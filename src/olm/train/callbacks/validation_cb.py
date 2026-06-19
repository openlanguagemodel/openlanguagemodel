"""
Validation callback for running validation during training.
"""

import torch
from typing import Optional

from olm.train.trainer import TrainerCallback


class ValidationCallback(TrainerCallback):
    """
    Callback to perform validation at specified intervals.

    Args:
        val_dataloader: Validation dataloader.
        eval_every: Validate every N steps.
        device: Device to run validation on.
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        val_dataloader,
        eval_every: int = 500,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.val_dataloader = val_dataloader
        self.eval_every = eval_every
        self.device = device
        self.use_amp = use_amp
        self.val_losses = []
        self.best_val_loss = float("inf")

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Run validation after each optimization step if needed."""
        if step % self.eval_every == 0:
            val_loss = self._validate(trainer)
            self.val_losses.append((step, val_loss))

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(
                    f"[Validation] Step {step}: New best validation loss: {val_loss:.4f}"
                )
            else:
                print(f"[Validation] Step {step}: Validation loss: {val_loss:.4f}")

            # Store in trainer state for other callbacks
            trainer.training_state["val_loss"] = val_loss
            trainer.training_state["is_best"] = is_best

    @torch.no_grad()
    def _validate(self, trainer) -> float:
        """Run validation loop."""
        trainer.model.eval()
        total_loss = 0.0
        num_batches = 0
        device = self.device or trainer.device
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        amp_enabled = self.use_amp and device_type == "cuda"

        for x, y in self.val_dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=amp_enabled):
                logits = trainer.model(x)
                loss = trainer.loss(logits, y)

            total_loss += loss.item()
            num_batches += 1

        trainer.model.train()
        return total_loss / max(num_batches, 1)
