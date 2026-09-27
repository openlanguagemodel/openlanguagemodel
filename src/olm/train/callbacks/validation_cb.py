"""
Validation callback for running validation during training.
"""

import torch
from typing import Any, Dict, Optional

from olm.core.dist import all_reduce, is_distributed, is_main_process
from olm.train.output import as_lm_output
from olm.train.trainer import TrainerCallback


class ValidationCallback(TrainerCallback):
    """
    Callback to perform validation at specified intervals.

    The validation loss is ``trainer.loss`` on the model's logits, averaged
    over validation batches; auxiliary and MTP losses are not included. Models
    may return a logits tensor, an ``LMOutput`` or a dict with ``logits``.
    Under DDP and FSDP every rank validates its share of the data and the
    result is averaged over all ranks, so every rank sees the same value.

    After each validation the callback sets ``val_loss``, ``val_step`` (the
    step it ran at) and ``is_best`` in ``trainer.training_state``. It runs
    before other callbacks (``order = -10``), so they see the new value in the
    same step.

    Args:
        val_dataloader: Validation dataloader.
        eval_every: Validate every N steps.
        device: Device to run validation on.
        use_amp: Whether to use automatic mixed precision.
    """

    order = -10

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
            if is_main_process():
                if is_best:
                    print(
                        f"[Validation] Step {step}: New best validation loss: {val_loss:.4f}"
                    )
                else:
                    print(f"[Validation] Step {step}: Validation loss: {val_loss:.4f}")

            # Store in trainer state for other callbacks
            trainer.training_state["val_loss"] = val_loss
            trainer.training_state["val_step"] = step
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
                logits = as_lm_output(trainer.model(x)).logits
                loss = trainer.loss(logits, y)

            total_loss += loss.item()
            num_batches += 1

        trainer.model.train()

        if is_distributed():
            totals = torch.tensor([total_loss, float(num_batches)], device=device)
            all_reduce(totals)
            total_loss, num_batches = totals[0].item(), int(totals[1].item())

        return total_loss / max(num_batches, 1)

    def state_dict(self) -> Dict[str, Any]:
        """Validation history and best loss, so ``is_best`` stays correct after a resume."""
        return {
            "val_losses": [list(entry) for entry in self.val_losses],
            "best_val_loss": self.best_val_loss,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the state returned by :meth:`state_dict`."""
        self.val_losses = [tuple(entry) for entry in state["val_losses"]]
        self.best_val_loss = state["best_val_loss"]
