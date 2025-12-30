from base import LossBase

import torch.nn.functional as F

class MaskedCELoss(LossBase):
    """
    Token-level cross-entropy with optional loss_mask.

    Expects:
      batch["logits"] : [B, T, V]
      batch["labels"] : [B, T]  (use ignore_index for tokens to ignore)
    Optional:
      batch["loss_mask"] : [B, T] (1/0 or bool)
    """
    name = "masked_ce"

    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index


    def forward(self, logits, y, mask):

        B, T, V = logits.shape

        per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=self.ignore_index,
            reduction="none",
        ).reshape(B, T)

        if mask is None:
            mask = (y != self.ignore_index)

        return self._apply_reduction(per_token, mask)
