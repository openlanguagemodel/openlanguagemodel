from base import LossBase
import torch

class ZLoss(LossBase):

    """
    Z-loss (logZ^2 penalty), commonly used as an auxiliary regularizer.

    For each token:
      logZ = logsumexp(logits, dim=-1)
      zloss = (logZ ** 2)

    Notes:
    - This does NOT include CE. Usually you add it to CE:
        total = ce_loss + z_coeff * z_loss
    """
    name = "z_loss"

    def __init__(self, z_coeff=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.z_coeff = z_coeff


    def forward(self, logits, y, mask=None):

        # logZ per token: [B, T]
        logZ = torch.logsumexp(logits, dim=-1)

        # per-token z-loss: [B, T]
        per_token = logZ * logZ

        z = self._apply_reduction(per_token, mask)

        return z * self.z_coeff