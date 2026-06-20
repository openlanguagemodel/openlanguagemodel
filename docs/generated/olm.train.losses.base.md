# `olm.train.losses.base`

Source: [`src/olm/train/losses/base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L1)

## Classes

### `LossBase(reduction='mean') -> None`

**Bases:** `Module`, `ABC`

Source: [`src/olm/train/losses/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L8)

Base class for all loss modules.

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor`

Source: [`src/olm/train/losses/base.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L32)

Apply loss to ``logits`` and ``y``.
