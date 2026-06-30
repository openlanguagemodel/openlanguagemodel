# `olm.train.losses.cross_entropy`

Source: [`src/olm/train/losses/cross_entropy.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L1)

## Classes

### `CrossEntropyLoss(reduction='mean') -> None`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/cross_entropy.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L6)

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/train/losses/cross_entropy.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L8)
