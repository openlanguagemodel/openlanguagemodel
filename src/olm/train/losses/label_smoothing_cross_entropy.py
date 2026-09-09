from olm.train.losses.base import LossBase
import torch
import torch.nn.functional as F
from olm.core.registry import LOSSES


@LOSSES.register("label_smoothing_cross_entropy")
class LabelSmoothingCrossEntropy(LossBase):
    """Cross entropy with label smoothing.

    Args:
        smoothing: Smoothing factor epsilon.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("Smoothing must be in [0, 1).")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (B, T, V)
            y: Tensor of shape (B, T)

        Returns:
            Scalar loss.
        """
        B, T, V = logits.shape

        logits = logits.view(-1, V)
        y = y.view(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        nll_loss = F.nll_loss(
            log_probs,
            y,
            ignore_index=-100,
            reduction="mean",
        )

        smooth_loss = -log_probs.mean(dim=-1)

        mask = (y != -100)
        smooth_loss = smooth_loss[mask].mean()

        return (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss