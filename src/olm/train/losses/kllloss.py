import torch
from base import LossBase

import torch.nn.functional as F


class KLLoss(LossBase):
    """
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
    """
    name = "kl_loss"

    def __init__(self, kl_coeff=1.0, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.kl_coeff = kl_coeff
        self.from_logits = from_logits

    def _compute(self, logits, ref, mask):

        if self.from_logits:
            logits = logits
            ref_logits = ref

            logp = F.log_softmax(logits, dim=-1)          # [B, T, V]
            p = torch.exp(logp)                           # [B, T, V]
            ref_logp = F.log_softmax(ref_logits, dim=-1)  # [B, T, V]
        else:
            logp = logits
            ref_logp = ref
            p = torch.exp(logp)

        # KL per token: sum_v p(v) * (logp(v) - ref_logp(v))  => [B, T]
        per_token = (p * (logp - ref_logp)).sum(dim=-1)

        kl = self._apply_reduction(per_token, mask)
        return kl * self.kl_coeff