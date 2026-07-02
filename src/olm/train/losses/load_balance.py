import torch
from olm.train.losses.base import LossBase
from olm.core.registry import LOSSES


@LOSSES.register("load_balance")
class LoadBalanceLoss(LossBase):
    """
    Auxiliary load-balancing loss for Mixture-of-Experts models.

    Encourages uniform expert utilisation by penalising the correlation
    between the fraction of tokens routed to each expert and the mean
    routing probability for that expert.

    ``L_balance = num_experts * sum_i( f_i * p_i )``

    where:
        - ``f_i`` = fraction of tokens assigned to expert *i* (top-k)
        - ``p_i`` = mean routing probability for expert *i*

    When routing is perfectly balanced, each ``f_i = 1/E`` and each
    ``p_i = 1/E``, giving a minimum loss of 1.0.

    Reference: Switch Transformer (arXiv:2101.03961), Section 3.

    Args:
        num_experts: Total number of routable experts.
        top_k: Number of experts selected per token.
        coefficient: Scaling factor on the loss (commonly 0.01 -- 0.001).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        coefficient: float = 0.01,
    ):
        super().__init__(reduction="mean")
        self.num_experts = num_experts
        self.top_k = top_k
        self.coefficient = coefficient

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            router_logits: Raw router logits ``[batch, seq_len, num_experts]``.
            top_k_indices: Selected expert indices ``[batch, seq_len, top_k]``.

        Returns:
            Scalar load-balance loss.
        """
        routing_probs = torch.softmax(router_logits.float(), dim=-1)
        mean_probs = routing_probs.mean(dim=[0, 1])  # [num_experts]

        one_hot = torch.zeros_like(routing_probs)
        one_hot.scatter_(-1, top_k_indices, 1.0)
        expert_fraction = one_hot.mean(dim=[0, 1])  # [num_experts]

        loss = self.num_experts * (expert_fraction * mean_probs).sum()
        return self.coefficient * loss
