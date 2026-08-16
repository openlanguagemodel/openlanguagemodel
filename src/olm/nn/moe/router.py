from dataclasses import dataclass
from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.torch_nn_wrappers import Linear


@dataclass
class MoERouterStats:
    """Routing metadata used by auxiliary losses and diagnostics."""

    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor
    router_logits: torch.Tensor
    expert_fraction: torch.Tensor
    mean_scores: torch.Tensor
    metadata: dict


class MoERouter(nn.Module):
    """
    Configurable Mixture-of-Experts router supporting multiple scoring and
    routing strategies used across modern MoE architectures.

    Supported scoring functions:
        - ``"softmax"``: Standard softmax over expert logits, then top-k.
        - ``"sigmoid"``: Sigmoid per-expert scores, then top-k. Used by
          Step 3.5 Flash, MiniMax M2.5, Sarvam, Ling 2.5.

    Supported routing strategies:
        - ``"topk"``: Standard top-k selection with re-normalized weights.
        - ``"noaux_tc"``: Auxiliary-loss-free routing with token-choice
          balancing via a learnable expert bias. Used by Sarvam, Ling 2.5,
          DeepSeek V3.

    Args:
        embed_dim: Hidden dimension of input tokens.
        num_experts: Total number of routable experts.
        top_k: Number of experts activated per token.
        scoring_func: ``"softmax"`` or ``"sigmoid"``.
        routing_method: ``"topk"`` or ``"noaux_tc"``.
        use_bias: Learnable bias added to router logits before scoring.
        norm_weights: Re-normalize top-k weights to sum to 1.
        fp32_gate: Cast logits to float32 before scoring for stability.
        routed_scaling_factor: Multiplicative factor applied to the
            combined expert output (e.g. 2.5 in Sarvam / Ling 2.5).
        expert_weight_norm: Normalize expert gate logits by expert output
            L2 norm (Step 3.5 Flash).
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 2,
        scoring_func: Literal["softmax", "sigmoid"] = "softmax",
        routing_method: Literal["topk", "noaux_tc"] = "topk",
        use_bias: bool = False,
        norm_weights: bool = True,
        fp32_gate: bool = False,
        routed_scaling_factor: float = 1.0,
        expert_weight_norm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.routing_method = routing_method
        self.norm_weights = norm_weights
        self.fp32_gate = fp32_gate
        self.routed_scaling_factor = routed_scaling_factor
        self.expert_weight_norm = expert_weight_norm

        self.gate = Linear(embed_dim, num_experts, bias=use_bias)

        if routing_method == "noaux_tc":
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.expert_bias = None
        self.last_stats: Optional[MoERouterStats] = None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to their top-k experts.

        Args:
            x: Hidden states ``[batch, seq_len, embed_dim]``.

        Returns:
            Tuple of:
                - ``top_k_indices``: ``[batch, seq_len, top_k]`` expert ids.
                - ``top_k_weights``: ``[batch, seq_len, top_k]`` routing weights.
                - ``router_logits``: ``[batch, seq_len, num_experts]`` raw logits
                  (for auxiliary loss computation).
        """
        logits = self.gate(x)

        if self.fp32_gate:
            logits = logits.float()

        router_logits = logits

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        selection_scores = scores
        if self.expert_bias is not None:
            selection_scores = scores + self.expert_bias

        top_k_weights, top_k_indices = torch.topk(
            selection_scores, self.top_k, dim=-1
        )

        if self.scoring_func == "sigmoid":
            top_k_weights = scores.gather(-1, top_k_indices)

        if self.norm_weights:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            top_k_weights = top_k_weights * self.routed_scaling_factor

        one_hot = torch.zeros_like(scores)
        one_hot.scatter_(-1, top_k_indices, 1.0)
        expert_fraction = one_hot.sum(dim=(0, 1)) / (
            x.shape[0] * x.shape[1] * self.top_k
        )
        mean_scores = scores.mean(dim=(0, 1))

        self.last_stats = MoERouterStats(
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            router_logits=router_logits,
            expert_fraction=expert_fraction,
            mean_scores=mean_scores,
            metadata={
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "scoring_func": self.scoring_func,
                "routing_method": self.routing_method,
            },
        )

        return top_k_indices, top_k_weights, router_logits

    @torch.no_grad()
    def update_expert_bias_(
        self,
        expert_fraction: Optional[torch.Tensor] = None,
        target_fraction: Optional[torch.Tensor] = None,
        update_rate: float = 1e-3,
    ) -> None:
        """
        Apply an auxiliary-loss-free expert-bias update.

        Over-used experts receive a lower bias; under-used experts receive a
        higher bias. This mirrors the mechanism used by loss-free balancing
        methods while keeping the update explicit and opt-in.
        """
        if self.expert_bias is None:
            raise ValueError("Expert bias updates require routing_method='noaux_tc'")

        if expert_fraction is None:
            if self.last_stats is None:
                raise ValueError("No router stats available for bias update")
            expert_fraction = self.last_stats.expert_fraction

        expert_fraction = expert_fraction.to(self.expert_bias.device)
        if target_fraction is None:
            target_fraction = torch.full_like(expert_fraction, 1.0 / self.num_experts)
        else:
            target_fraction = target_fraction.to(self.expert_bias.device)

        self.expert_bias.add_(update_rate * (target_fraction - expert_fraction))
