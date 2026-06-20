# `olm.train.losses.kllloss`

Source: [`src/olm/train/losses/kllloss.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/losses/kllloss.py#L1)

## Classes

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
