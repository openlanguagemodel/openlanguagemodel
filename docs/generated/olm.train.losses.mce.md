# `olm.train.losses.mce`

Source: [`src/olm/train/losses/mce.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/mce.py#L1)

## Classes

### `MaskedCELoss(ignore_index=-100, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/mce.py:5`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/mce.py#L5)

Token-level cross-entropy with optional loss_mask.

Expects:
  batch["logits"] : [B, T, V]
  batch["labels"] : [B, T]  (use ignore_index for tokens to ignore)
Optional:
  batch["loss_mask"] : [B, T] (1/0 or bool)

#### Methods

##### `forward(self, logits, y, mask)`

Source: [`src/olm/train/losses/mce.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/mce.py#L22)
