# `olm.train.callbacks.early_stopping_cb`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L1)

Early stopping callback to prevent overfitting.

## Classes

### `EarlyStoppingCallback(patience: int = 5, min_delta: float = 0.0)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L8)

Callback to stop training early if validation loss doesn't improve.

Args:
    patience: Number of validation checks to wait for improvement.
    min_delta: Minimum change in validation loss to qualify as improvement.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/early_stopping_cb.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/early_stopping_cb.py#L24)

Check for early stopping after each step.
