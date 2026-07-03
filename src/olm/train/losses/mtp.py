import torch
import torch.nn.functional as F
from typing import List

from olm.train.losses.base import LossBase
from olm.core.registry import LOSSES


@LOSSES.register("mtp")
class MTPLoss(LossBase):
    """
    Multi-Token Prediction auxiliary loss.

    Computes a weighted sum of per-position cross-entropy losses from the
    MTP heads.  Each head *i* predicts the token at position ``t + i + 1``
    and contributes ``weight_i * CE(logits_i, target_{t+i+1})``.

    Typically the main model's next-token loss is computed separately and
    this loss is added with a coefficient.

    Args:
        num_heads: Number of MTP prediction heads.
        head_weights: Per-head loss weight.  If ``None``, all heads are
            weighted equally as ``1 / num_heads``.
        coefficient: Global scaling factor applied to the combined loss.
    """

    def __init__(
        self,
        num_heads: int = 3,
        head_weights: List[float] = None,
        coefficient: float = 1.0,
    ):
        super().__init__(reduction="mean")
        self.num_heads = num_heads
        self.coefficient = coefficient

        if head_weights is None:
            self.head_weights = [1.0 / num_heads] * num_heads
        else:
            assert len(head_weights) == num_heads
            self.head_weights = head_weights

    def forward(
        self,
        mtp_logits: List[torch.Tensor],
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            mtp_logits: List of ``num_heads`` tensors, each
                ``[batch, seq_len_i, vocab_size]`` where ``seq_len_i``
                decreases with head index.
            labels: Full target sequence ``[batch, seq_len]``.

        Returns:
            Scalar MTP loss.
        """
        total_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)

        for i, logits in enumerate(mtp_logits):
            offset = i + 2
            if offset > labels.shape[1]:
                break

            target = labels[:, offset:]
            seq_len = min(logits.shape[1], target.shape[1])
            logits_i = logits[:, :seq_len]
            target_i = target[:, :seq_len]

            loss_i = F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                target_i.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            total_loss = total_loss + self.head_weights[i] * loss_i

        return self.coefficient * total_loss
