"""
Checkpoint callback for saving model checkpoints during training.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from olm.core.dist import is_distributed, is_main_process
from olm.train.trainer import TrainerCallback


class CheckpointCallback(TrainerCallback):
    """
    Callback to save training checkpoints at specified intervals.

    Every file is a complete training checkpoint written by
    ``trainer.save_checkpoint`` (format version 1, see
    ``olm.train.checkpointing``), so any of them can resume a run with
    ``Trainer(..., resume_from=path)``. Files are written atomically. Under DDP
    every rank takes part in each save and rank 0 writes the file.

    Files in ``checkpoint_dir``:
        ``step_<N>.pt`` every ``save_every`` optimizer steps and when training
        ends; only the newest ``keep_last_n`` are kept.
        ``best_model.pt`` whenever a validation improves the monitored metric.
        It is never deleted.

    The callback runs after the other callbacks of a step (``order = 10``), so
    a checkpoint holds this step's validation result and early-stopping state.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_every: Save checkpoint every N steps.
        keep_last_n: Keep only the last N checkpoints.
        save_best: Whether to save the best model based on validation loss.
        monitor: Key in ``trainer.training_state`` that decides the best model.
            Read once per validation, which is detected by a new
            ``training_state["val_step"]``. Defaults to ``"val_loss"``.
        mode: ``"min"`` when a lower ``monitor`` value is better, ``"max"``
            when a higher one is. Defaults to ``"min"``.
        save_on_train_end: Also save ``step_<N>.pt`` for the last step when
            training ends, unless it was just saved. Defaults to True.
    """

    order = 10

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1000,
        keep_last_n: int = 5,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_on_train_end: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_on_train_end = save_on_train_end

        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None
        self._last_val_step: Optional[int] = None
        self._last_saved_step: Optional[int] = None
        self._train_begin_step = 0

    def on_train_begin(self, trainer) -> None:
        """Remember where this run started, for the final save."""
        self._train_begin_step = trainer.global_step

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Save checkpoint after each optimization step if needed."""
        # Update the best value first, so a step checkpoint written in the same
        # step already knows about this validation.
        is_best = self.save_best and self._is_new_best(trainer)

        if step % self.save_every == 0:
            self._save_checkpoint(trainer, step, is_regular=True)

        if is_best:
            self._save_checkpoint(trainer, step, is_best=True)

    def on_train_end(self, trainer) -> None:
        """Save the final step, so the end of a run is never lost."""
        step = trainer.global_step
        if (
            self.save_on_train_end
            and step > self._train_begin_step
            and step != self._last_saved_step
        ):
            self._save_checkpoint(trainer, step, is_regular=True)

    def _is_new_best(self, trainer) -> bool:
        """Whether a validation finished in this step and improved ``monitor``."""
        state = trainer.training_state
        val_step = state.get("val_step")
        if val_step is None:
            # Validation callbacks written before val_step existed only set
            # is_best; honour it once.
            if state.get("is_best"):
                state["is_best"] = False
                return True
            return False

        if val_step == self._last_val_step:
            return False
        self._last_val_step = val_step

        value = state.get(self.monitor)
        if value is None:
            return False
        if self.best_value is None or (
            value < self.best_value if self.mode == "min" else value > self.best_value
        ):
            self.best_value = float(value)
            self.best_step = val_step
            return True
        return False

    def _save_checkpoint(
        self, trainer, step: int, is_regular: bool = False, is_best: bool = False
    ) -> None:
        """Save a checkpoint."""
        if not hasattr(trainer, "save_checkpoint"):
            self._save_legacy_checkpoint(trainer, step, is_regular, is_best)
            return

        if is_regular:
            checkpoint_path = self.checkpoint_dir / f"step_{step}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            self._last_saved_step = step
            if is_main_process():
                print(f"[Checkpoint] Saved checkpoint: {checkpoint_path}", flush=True)
                self._cleanup_old_checkpoints()

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(str(best_path))
            if is_main_process():
                print(f"[Checkpoint] Saved best model: {best_path}", flush=True)

    def _save_legacy_checkpoint(
        self, trainer, step: int, is_regular: bool, is_best: bool
    ) -> None:
        """Pre-version-1 layout, for trainer objects without ``save_checkpoint``."""
        if is_distributed() and not is_main_process():
            return

        checkpoint = {
            "step": step,
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "losses": trainer.losses,
        }
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        if is_regular:
            checkpoint_path = self.checkpoint_dir / f"step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            self._last_saved_step = step
            print(f"[Checkpoint] Saved checkpoint: {checkpoint_path}")
            self._cleanup_old_checkpoints()

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model: {best_path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the most recent numbered checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[: -self.keep_last_n]:
                old_ckpt.unlink()
                print(f"[Checkpoint] Removed old checkpoint: {old_ckpt}")

    def state_dict(self) -> Dict[str, Any]:
        """Best value so far, so a resumed run does not overwrite a better model."""
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "last_val_step": self._last_val_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the state returned by :meth:`state_dict`."""
        self.best_value = state.get("best_value")
        self.best_step = state.get("best_step")
        self._last_val_step = state.get("last_val_step")
