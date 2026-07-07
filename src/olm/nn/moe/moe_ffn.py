from typing import Type, Literal, Optional
import torch
import torch.nn as nn

from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.moe.router import MoERouter, MoERouterStats


class MoEFeedForward(FeedForwardBase):
    """
    Mixture-of-Experts feed-forward layer with configurable routing.

    Extends ``FeedForwardBase`` with support for sigmoid routing,
    auxiliary-loss-free balancing, routed scaling, shared experts, and
    other features required by Step 3.5 Flash, MiniMax M2.5, Sarvam,
    Ling 2.5, and Qwen3.5.

    Args:
        embed_dim: Input / output hidden dimension.
        expert_cls: FFN class to instantiate for each expert.
        num_experts: Total number of routable experts.
        num_shared_experts: Always-active experts that process every token.
        top_k: Experts selected per token.
        expert_kwargs: Keyword arguments forwarded to ``expert_cls``.
        scoring_func: ``"softmax"`` or ``"sigmoid"``.
        routing_method: ``"topk"`` or ``"noaux_tc"``.
        use_router_bias: Learnable bias on router logits.
        norm_weights: Re-normalize top-k weights.
        fp32_gate: Run gate in FP32.
        routed_scaling_factor: Scale factor on routed output.
    """

    def __init__(
        self,
        embed_dim: int,
        expert_cls: Type[nn.Module],
        num_experts: int = 8,
        num_shared_experts: int = 0,
        top_k: int = 2,
        expert_kwargs: Optional[dict] = None,
        scoring_func: Literal["softmax", "sigmoid"] = "softmax",
        routing_method: Literal["topk", "noaux_tc"] = "topk",
        use_router_bias: bool = False,
        norm_weights: bool = True,
        fp32_gate: bool = False,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__(embed_dim)
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.last_router_stats: Optional[MoERouterStats] = None

        expert_kwargs = expert_kwargs or {}

        self.router = MoERouter(
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=scoring_func,
            routing_method=routing_method,
            use_bias=use_router_bias,
            norm_weights=norm_weights,
            fp32_gate=fp32_gate,
            routed_scaling_factor=routed_scaling_factor,
        )

        self.experts = nn.ModuleList(
            [expert_cls(embed_dim, **expert_kwargs) for _ in range(num_experts)]
        )

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [expert_cls(embed_dim, **expert_kwargs) for _ in range(num_shared_experts)]
            )
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MoE routing.

        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            Tuple of:
                - output: ``[batch, seq_len, embed_dim]``
                - router_logits: ``[batch, seq_len, num_experts]`` for loss.
        """
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)

        shared_output = torch.zeros_like(x)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x)

        top_k_indices, top_k_weights, router_logits = self.router(x)
        self.last_router_stats = self.router.last_stats
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_mask = top_k_indices_flat == i
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x_flat[token_mask]
            expert_out = expert(expert_input)

            relevant_weights = (top_k_weights_flat * expert_mask.float()).sum(dim=-1)
            selected_weights = relevant_weights[token_mask].unsqueeze(-1)

            final_output[token_mask] += expert_out * selected_weights

        final_output = final_output.view(batch_size, seq_len, embed_dim)
        final_output = final_output + shared_output

        return final_output, router_logits

    def get_router_stats(self) -> Optional[MoERouterStats]:
        """Return routing stats from the most recent forward pass."""
        return self.last_router_stats
