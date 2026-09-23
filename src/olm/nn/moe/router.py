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
        num_expert_groups: Number of logical expert groups.
        top_k_groups: Number of groups made eligible before expert top-k.
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
        num_expert_groups: int = 1,
        top_k_groups: int = 1,
        expert_weight_norm: bool = False,
    ):
        super().__init__()
        if num_experts % num_expert_groups != 0:
            raise ValueError("num_experts must be divisible by num_expert_groups")
        if not 1 <= top_k_groups <= num_expert_groups:
            raise ValueError("top_k_groups must be in [1, num_expert_groups]")

        experts_per_group = num_experts // num_expert_groups
        if top_k > top_k_groups * experts_per_group:
            raise ValueError("top_k exceeds the experts available in selected groups")

        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.routing_method = routing_method
        self.norm_weights = norm_weights
        self.fp32_gate = fp32_gate
        self.routed_scaling_factor = routed_scaling_factor
        self.num_expert_groups = num_expert_groups
        self.top_k_groups = top_k_groups
        self.expert_weight_norm = expert_weight_norm

        self.gate = Linear(embed_dim, num_experts, bias=use_bias)

        if routing_method == "noaux_tc":
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
            self.register_buffer("expert_bias_momentum", torch.zeros(num_experts))
        else:
            self.expert_bias = None
            self.register_buffer("expert_bias_momentum", None)
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
        if self.fp32_gate:
            bias = self.gate.bias
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = F.linear(
                    x.float(),
                    self.gate.weight.float(),
                    None if bias is None else bias.float(),
                )
        else:
            logits = self.gate(x)

        router_logits = logits

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        selection_scores = scores
        if self.expert_bias is not None:
            selection_scores = scores + self.expert_bias

        if self.num_expert_groups > 1:
            experts_per_group = self.num_experts // self.num_expert_groups
            grouped_scores = selection_scores.view(
                *selection_scores.shape[:-1],
                self.num_expert_groups,
                experts_per_group,
            )
            group_scores = grouped_scores.topk(
                k=min(2, experts_per_group), dim=-1
            ).values.sum(dim=-1)
            selected_groups = group_scores.topk(self.top_k_groups, dim=-1).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(-1, selected_groups, True)
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped_scores)
            selection_scores = grouped_scores.masked_fill(
                ~expert_mask, float("-inf")
            ).reshape_as(selection_scores)

        top_k_weights, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)

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
                "num_expert_groups": self.num_expert_groups,
                "top_k_groups": self.top_k_groups,
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

    @torch.no_grad()
    def update_expert_bias_softclamped_momentum_(
        self,
        expert_fraction: Optional[torch.Tensor] = None,
        update_rate: float = 5e-4,
        momentum: float = 0.5,
        clamp_scale: float = 2.0,
    ) -> None:
        """Update loss-free routing biases with soft-clamped momentum.

        The normalized load error is centered across experts, bounded with
        ``tanh``, and smoothed before being applied to the selection bias.
        This is the SMEBU update used by Arcee Trinity.

        Args:
            expert_fraction: Observed fraction of routed assignments per expert.
                Uses the most recent router statistics when omitted.
            update_rate: Maximum update magnitude before momentum smoothing.
            momentum: Fraction of the previous update retained.
            clamp_scale: Multiplier applied before the ``tanh`` soft clamp.
        """
        if self.expert_bias is None or self.expert_bias_momentum is None:
            raise ValueError("SMEBU updates require routing_method='noaux_tc'")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if update_rate < 0.0:
            raise ValueError(f"update_rate must be non-negative, got {update_rate}")

        if expert_fraction is None:
            if self.last_stats is None:
                raise ValueError("No router stats available for SMEBU update")
            expert_fraction = self.last_stats.expert_fraction

        expert_fraction = expert_fraction.to(self.expert_bias.device)
        target = torch.full_like(expert_fraction, 1.0 / self.num_experts)
        violation = (target - expert_fraction) / target.clamp_min(1e-12)
        update = update_rate * torch.tanh(clamp_scale * violation)
        update = update - update.mean()

        self.expert_bias_momentum.mul_(momentum).add_(update, alpha=1.0 - momentum)
        self.expert_bias.add_(self.expert_bias_momentum)
