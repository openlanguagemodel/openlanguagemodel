# `olm.train.callbacks.early_stopping_cb`

Early stopping callback to prevent overfitting.

## Classes

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

Callback to stop training early if validation loss doesn't improve.

Args:
    patience: Number of validation checks to wait for improvement.
    min_delta: Minimum change in validation loss to qualify as improvement.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Check for early stopping after each step.
