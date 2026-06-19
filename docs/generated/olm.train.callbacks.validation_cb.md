# `olm.train.callbacks.validation_cb`

Validation callback for running validation during training.

## Classes

### `ValidationCallback(val_dataloader, eval_every: int = 500, device: str | None = None, use_amp: bool = True)`

Callback to perform validation at specified intervals.

Args:
    val_dataloader: Validation dataloader.
    eval_every: Validate every N steps.
    device: Device to run validation on.
    use_amp: Whether to use automatic mixed precision.

#### Methods

- `on_step_end(self, trainer, step: int, loss: float) -> None`
  Run validation after each optimization step if needed.
