# `olm.train.losses.zloss`

Source: [`src/olm/train/losses/zloss.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/zloss.py#L1)

## Classes

### `ZLoss(z_coeff=0.0001, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/zloss.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/zloss.py#L4)

Z-loss (logZ^2 penalty), commonly used as an auxiliary regularizer.

For each token:
  logZ = logsumexp(logits, dim=-1)
  zloss = (logZ ** 2)

Notes:
- This does NOT include CE. Usually you add it to CE:
    total = ce_loss + z_coeff * z_loss

#### Methods

##### `forward(self, logits, y, mask=None)`

Source: [`src/olm/train/losses/zloss.py:24`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/zloss.py#L24)
