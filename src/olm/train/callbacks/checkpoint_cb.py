"""
Checkpoint callback for saving model checkpoints during training.
"""

from pathlib import Path
import torch

from olm.train.trainer import TrainerCallback


class CheckpointCallback(TrainerCallback):
    """
    Callback to save model checkpoints at specified intervals.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_every: Save checkpoint every N steps.
        keep_last_n: Keep only the last N checkpoints.
        save_best: Whether to save the best model based on validation loss.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1000,
        keep_last_n: int = 5,
        save_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_best = save_best

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Save checkpoint after each optimization step if needed."""
        if step % self.save_every == 0:
            self._save_checkpoint(trainer, step, is_regular=True)

        # Save best model if validation was just performed
        if self.save_best and "is_best" in trainer.training_state:
            if trainer.training_state["is_best"]:
                self._save_checkpoint(trainer, step, is_best=True)
                trainer.training_state["is_best"] = False  # Reset flag

    def _save_checkpoint(
        self, trainer, step: int, is_regular: bool = False, is_best: bool = False
    ) -> None:
        """Save a checkpoint."""
        checkpoint = {
            "step": step,
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "losses": trainer.losses,
        }

        # Add scheduler state if present
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Save regular checkpoint
        if is_regular:
            checkpoint_path = self.checkpoint_dir / f"step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"[Checkpoint] Saved checkpoint: {checkpoint_path}")

            # Keep only last N checkpoints (sort numerically, not alphabetically)
            checkpoints = sorted(
                self.checkpoint_dir.glob("step_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if len(checkpoints) > self.keep_last_n:
                for old_ckpt in checkpoints[: -self.keep_last_n]:
                    old_ckpt.unlink()
                    print(f"[Checkpoint] Removed old checkpoint: {old_ckpt}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model: {best_path}")
