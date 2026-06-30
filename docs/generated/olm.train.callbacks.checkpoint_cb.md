# `olm.train.callbacks.checkpoint_cb`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L1)

Checkpoint callback for saving model checkpoints during training.

## Classes

### `CheckpointCallback(checkpoint_dir: str = 'checkpoints', save_every: int = 1000, keep_last_n: int = 5, save_best: bool = True)`

**Bases:** `olm.train.trainer.trainer.TrainerCallback`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:12`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L12)

Callback to save model checkpoints at specified intervals.

**Parameters**

- `checkpoint_dir`: Directory to save checkpoints.
- `save_every`: Save checkpoint every N steps.
- `keep_last_n`: Keep only the last N checkpoints.
- `save_best`: Whether to save the best model based on validation loss.

#### Methods

##### `on_step_end(self, trainer, step: int, loss: float) -> None`

Source: [`src/olm/train/callbacks/checkpoint_cb.py:36`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/callbacks/checkpoint_cb.py#L36)

Save checkpoint after each optimization step if needed.
