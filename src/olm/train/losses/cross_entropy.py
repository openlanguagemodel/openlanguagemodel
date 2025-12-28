from olm.train.losses.base import LossBase
import torch
from olm.core.registry import LOSSES


@LOSSES.register("cross_entropy")
class CrossEntropyLoss(LossBase):
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction="mean",
        )