"""
Early stopping callback to prevent overfitting.
"""

import warnings
from typing import Any, Dict, Optional

from olm.train.trainer import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback to stop training early if a validation metric doesn't improve.

    Patience counts validations, not optimizer steps: training stops after
    ``patience`` validations in a row that fail to improve on the best value
    by more than ``min_delta``. A validation is recognised by a new
    ``trainer.training_state["val_step"]``, which ``ValidationCallback``
    writes together with ``val_loss``; a custom validation callback must
    write both. The best value and the count survive a resume.

    Args:
        patience: Number of validation checks to wait for improvement.
        min_delta: Minimum change in the monitored metric to qualify as improvement.
        monitor: Key of the metric in ``trainer.training_state``. Defaults to
            ``"val_loss"``.
        mode: ``"min"`` when lower is better (losses), ``"max"`` when higher is
            better (accuracies). Defaults to ``"min"``.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_step: Optional[int] = None
        self.wait = 0
        self.stopped_step = 0
        self._last_val_step: Optional[int] = None
        self._warned_missing_val_step = False

    @property
    def best_loss(self) -> float:
        """The best monitored value so far (the name predates ``monitor``)."""
        return self.best_value

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        """Count a validation that finished in this step."""
        state = trainer.training_state
        val_step = state.get("val_step")
        if val_step is None:
            if self.monitor in state and not self._warned_missing_val_step:
                warnings.warn(
                    f"EarlyStoppingCallback sees {self.monitor!r} but no "
                    "training_state['val_step'], so it cannot tell a new "
                    "validation from an old one and never stops. Validation "
                    "callbacks must set val_step to the step they ran at.",
                    stacklevel=2,
                )
                self._warned_missing_val_step = True
            return

        if val_step == self._last_val_step:
            return
        self._last_val_step = val_step

        value = state.get(self.monitor)
        if value is None:
            return

        if self._improved(value):
            self.best_value = float(value)
            self.best_step = val_step
            self.wait = 0
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_step = step
            print(
                f"[Early Stopping] Stopping at step {step}: {self.monitor} has not "
                f"improved on {self.best_value:.4f} (step {self.best_step}) for "
                f"{self.wait} validations",
                flush=True,
            )
            # Signal to stop training (trainer needs to handle this)
            trainer.training_state["should_stop"] = True

    def state_dict(self) -> Dict[str, Any]:
        """Best value and patience count, so a resumed run continues counting."""
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "wait": self.wait,
            "stopped_step": self.stopped_step,
            "last_val_step": self._last_val_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the state returned by :meth:`state_dict`."""
        self.best_value = state["best_value"]
        self.best_step = state.get("best_step")
        self.wait = state["wait"]
        self.stopped_step = state.get("stopped_step", 0)
        self._last_val_step = state.get("last_val_step")
