from typing import Optional

import torch
import torch.nn.functional as F

from olm.train.losses.base import LossBase


class MaskedCELoss(LossBase):
    """
    Token-level cross-entropy with an optional per-token loss mask.

    Positions whose label is ``ignore_index`` never count. ``mask`` removes
    more positions, or weights them, for example to leave prompt tokens out of
    a supervised fine-tuning loss.

    Args:
        ignore_index (int): Label value of positions without a target.
        reduction (str): ``"mean"`` (average over the counted positions),
            ``"sum"`` or ``"none"`` (per-token losses, shaped like ``y``).

    Forward:
        Takes ``logits`` shaped ``[batch, seq_len, vocab_size]``, labels ``y``
        shaped ``[batch, seq_len]`` and an optional ``mask`` of the same shape
        (1/0, bool or float weights). With ``mask=None`` every position whose
        label is not ``ignore_index`` counts, so the Trainer can call it as
        ``loss(logits, labels)``, like ``CrossEntropyLoss``.
    """

    name = "masked_ce"

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, V = logits.shape

        per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=self.ignore_index,
            reduction="none",
        ).reshape(B, T)

        valid = y != self.ignore_index
        if mask is None:
            mask = valid
        else:
            mask = mask.to(per_token.dtype) * valid

        return self._apply_reduction(per_token, mask)
