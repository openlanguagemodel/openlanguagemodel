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
          DeepSeek V3, Nemotron 3.

    Experts may additionally be partitioned into ``n_group`` groups, in which
    case only the ``topk_group`` best-scoring groups are eligible for
    selection (group-limited routing, used by DeepSeek V3 and Nemotron 3).

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
        n_group: Number of expert groups for group-limited routing. ``None``
            (the default) considers every expert.
        topk_group: Number of groups a token may draw experts from. Required
            when ``n_group`` is set.
        group_score_top_k: How many of a group's best expert scores are summed
            to score that group. Defaults to 2, as in DeepSeek V3 / Nemotron 3.
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
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        group_score_top_k: int = 2,
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

        if n_group is not None:
            if topk_group is None:
                raise ValueError("topk_group is required when n_group is set")
            if num_experts % n_group != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by "
                    f"n_group ({n_group})"
                )
            if not 1 <= topk_group <= n_group:
                raise ValueError(
                    f"topk_group ({topk_group}) must be in [1, n_group={n_group}]"
                )
        self.n_group = n_group
        self.topk_group = topk_group
        self.group_score_top_k = group_score_top_k

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

        if self.n_group is not None:
            selection_scores = self._mask_unselected_groups(selection_scores)

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
                "n_group": self.n_group,
                "topk_group": self.topk_group,
            },
        )

        return top_k_indices, top_k_weights, router_logits

    def _mask_unselected_groups(self, selection_scores: torch.Tensor) -> torch.Tensor:
        """
        Restrict selection to the ``topk_group`` best expert groups.

        Each group is scored by the sum of its ``group_score_top_k`` highest
        expert scores; experts outside the winning groups are masked out so
        ``topk`` can never pick them.

        Args:
            selection_scores: ``[batch, seq_len, num_experts]`` scores that
                top-k selection will run on.

        Returns:
            The same tensor with non-eligible experts set to ``-inf``.
        """
        *leading, num_experts = selection_scores.shape
        experts_per_group = num_experts // self.n_group
        grouped = selection_scores.view(*leading, self.n_group, experts_per_group)

        group_scores = grouped.topk(
            min(self.group_score_top_k, experts_per_group), dim=-1
        ).values.sum(dim=-1)
        group_indices = torch.topk(group_scores, self.topk_group, dim=-1).indices

        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_indices, True)
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape(
            *leading, num_experts
        )

        return selection_scores.masked_fill(~expert_mask, float("-inf"))

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
