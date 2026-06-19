# `olm.train.callbacks.validation_cb`

Source: [`src/olm/train/callbacks/validation_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L1)

Validation callback for running validation during training.

## Classes

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/validation_cb.py:11`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L11)

Callback to perform validation at specified intervals.

Args:
    val_dataloader: Validation dataloader.
    eval_every: Validate every N steps.
    device: Device to run validation on.
    use_amp: Whether to use automatic mixed precision.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/validation_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/validation_cb.py#L36)

Run validation after each optimization step if needed.
