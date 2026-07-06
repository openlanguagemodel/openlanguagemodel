from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.torch_nn_wrappers import Linear


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

        return top_k_indices, top_k_weights, router_logits
