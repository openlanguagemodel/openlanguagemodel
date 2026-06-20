# `olm.train.losses`

Source: [`src/olm/train/losses/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/__init__.py#L1)

Loss functions for OLM training.

## Classes

### `CrossEntropyLoss(reduction='mean') -> None`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/cross_entropy.py:6`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L6)

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/train/losses/cross_entropy.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/cross_entropy.py#L8)

### `KLLoss(kl_coeff=1.0, from_logits=True, **kwargs)`

**Bases:** `olm.train.losses.base.LossBase`

Source: [`src/olm/train/losses/kllloss.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/kllloss.py#L7)

Forward KL penalty between a policy distribution and a reference distribution.

Supports two input modes:
  1) From logits:
     logits      : [B, T, V]  (policy logits)
     ref  : [B, T, V]  (reference logits)

  2) From log-probs:
     logp       : [B, T, V]  (policy log-probs)
     ref_logp    : [B, T, V]  (reference log-probs)

Optional:
  loss_mask : [B, T] (bool or 0/1) to mask tokens

Output:
  scalar loss (unless reduction="none") = kl_coeff * reduced(KL per token)

Notes:
  - This computes the *full-distribution* KL per token (sums over vocab V).
  - That's the common KL regularizer used to keep the policy close to a reference.

#### Methods

##### `forward(self, logits, ref, mask=None)`

Source: [`src/olm/train/losses/kllloss.py:57`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/kllloss.py#L57)

### `LossBase(reduction='mean') -> None`

**Bases:** `Module`, `ABC`

Source: [`src/olm/train/losses/base.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L8)

Base class for all loss modules.

#### Methods

##### `forward(self, logits: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor`

Source: [`src/olm/train/losses/base.py:32`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/base.py#L32)

Apply loss to ``logits`` and ``y``.

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
