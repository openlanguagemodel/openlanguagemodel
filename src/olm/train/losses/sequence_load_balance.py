from typing import Literal

import torch
import torch.nn.functional as F

from olm.core.registry import LOSSES
from olm.train.losses.base import LossBase


@LOSSES.register("sequence_load_balance")
class SequenceLoadBalanceLoss(LossBase):
    """
    Sequence-wise MoE load-balancing loss.

    Computes a Switch-style balance penalty independently per sequence, then
    averages across the batch. This is useful for modern MoE models that use a
    small complementary sequence-wise balance term alongside loss-free routing.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        coefficient: float = 0.001,
        scoring_func: Literal["softmax", "sigmoid"] = "softmax",
    ):
        super().__init__(reduction="mean")
        self.num_experts = num_experts
        self.top_k = top_k
        self.coefficient = coefficient
        self.scoring_func = scoring_func

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.scoring_func == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
            routing_probs = routing_scores / routing_scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        else:
            routing_probs = F.softmax(router_logits.float(), dim=-1)

        one_hot = torch.zeros_like(routing_probs)
        one_hot.scatter_(-1, top_k_indices, 1.0)

        seq_len = router_logits.shape[1]
        expert_fraction = one_hot.sum(dim=1) / max(seq_len * self.top_k, 1)
        mean_probs = routing_probs.mean(dim=1)

        loss = self.num_experts * (expert_fraction * mean_probs).sum(dim=-1)
        return self.coefficient * loss.mean()
