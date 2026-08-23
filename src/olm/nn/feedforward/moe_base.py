from typing import Optional, List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.moe.router import MoERouter as ConfigurableMoERouter
from olm.nn.torch_nn_wrappers import Linear

class MoERouter(nn.Module):
    """
    Minimal softmax router for Mixture of Experts.

    Routes input tokens to the top-k experts based on learned gate logits.
    ``MoEFeedForwardBase`` routes through ``olm.nn.moe.MoERouter`` instead,
    which additionally covers sigmoid scoring, correction-bias balancing and
    group-limited routing; this class is kept as the smallest readable
    reference implementation.

    Args:
        embed_dim (int): Dimension the router operates on.
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts each token is routed to.
        routed_scaling_factor (float, optional): Multiplier applied to the
            re-normalized top-k weights. Several MoE models (DeepSeek-V3,
            Sarvam, Nemotron-H) scale the routed branch by a constant so it
            keeps parity with an always-active shared expert. Defaults to 1.0
            (no scaling).
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        top_k: int = 2,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.gate = Linear(embed_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route each token to its top-k experts.

        Args:
            x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Expert indices and normalized
            routing weights, both shaped ``[batch, seq_len, top_k]``.
        """
        # x: (batch_size, seq_len, embed_dim)
        logits = self.gate(x) # (batch, seq, num_experts)

        # Calculate routing weights
        weights = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)

        # Re-normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            top_k_weights = top_k_weights * self.routed_scaling_factor

        return top_k_indices, top_k_weights

class MoEFeedForwardBase(FeedForwardBase):
    """
    Base class for Mixture of Experts FeedForward networks.

    Supports:
    - Configurable routing through ``olm.nn.moe.MoERouter`` -- softmax or
      sigmoid scoring, auxiliary-loss-free (``noaux_tc``) correction bias,
      group-limited routing, and a constant routed scaling factor
    - Shared experts (always active), which may use their own width and
      their own dimension
    - A router dimension independent of the expert dimension
    - Dynamic expert instantiation
    """
    def __init__(
        self,
        embed_dim: int,
        expert_cls: Type[nn.Module],
        num_experts: int = 8,
        num_shared_experts: int = 0,
        top_k: int = 2,
        expert_kwargs: dict = None,
        shared_expert_kwargs: dict = None,
        shared_embed_dim: int = None,
        router_embed_dim: int = None,
        router_kwargs: dict = None,
        routed_scaling_factor: float = 1.0,
        **kwargs
    ):
        """
        Args:
            embed_dim: Input/output dimension.
            expert_cls: The class of the expert feedforward network (e.g. ClassicFFN).
            num_experts: Total number of routable experts.
            num_shared_experts: Number of shared experts that process every token.
            top_k: Number of experts to route to for each token.
            expert_kwargs: Arguments to pass to the expert constructor.
            shared_expert_kwargs: Arguments for the shared-expert constructor.
                Defaults to ``expert_kwargs``. Models whose shared experts are
                wider than their routed experts (e.g. Nemotron-H) override
                ``hidden_dim`` here.
            shared_embed_dim: Dimension the shared experts operate on. Defaults
                to ``embed_dim``. Differs only when the experts run in a
                compressed space while the shared branch stays full-width
                (see ``LatentMoEFFN``).
            router_embed_dim: Dimension the router operates on. Defaults to
                ``embed_dim``. Differs when routing decisions are taken on the
                full-width hidden state while the experts run in a compressed
                space (see ``LatentMoEFFN``).
            router_kwargs: Routing options forwarded to ``olm.nn.moe.MoERouter``
                -- e.g. ``scoring_func``, ``routing_method``, ``n_group``,
                ``topk_group``, ``fp32_gate``. Defaults to plain softmax top-k.
            routed_scaling_factor: Constant multiplier on the routing weights.
            **kwargs: Additional arguments passed to FeedForwardBase.
        """
        super().__init__(embed_dim)
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_kwargs = expert_kwargs or {}
        self.shared_expert_kwargs = (
            self.expert_kwargs if shared_expert_kwargs is None else shared_expert_kwargs
        )
        self.shared_embed_dim = embed_dim if shared_embed_dim is None else shared_embed_dim
        self.router_embed_dim = embed_dim if router_embed_dim is None else router_embed_dim

        # Initialize Router. The shared, configurable router covers softmax and
        # sigmoid scoring, correction-bias (``noaux_tc``) balancing and
        # group-limited routing, so architectures do not need their own.
        self.router = ConfigurableMoERouter(
            embed_dim=self.router_embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            **(router_kwargs or {}),
        )
        self.last_router_logits: Optional[torch.Tensor] = None

        # Initialize Routable Experts
        self.experts = nn.ModuleList([
            expert_cls(embed_dim, **self.expert_kwargs)
            for _ in range(num_experts)
        ])

        # Initialize Shared Experts (if any)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                expert_cls(self.shared_embed_dim, **self.shared_expert_kwargs)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def compute_shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum of the always-active shared experts.

        Args:
            x (torch.Tensor): Hidden states shaped ``[batch, seq_len, shared_embed_dim]``.

        Returns:
            torch.Tensor: Same shape as ``x``; zeros when there are no shared experts.
        """
        if self.shared_experts is None:
            return torch.zeros_like(x)

        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)
        return shared_output

    def compute_routed(
        self, x: torch.Tensor, router_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Route each token to its top-k experts and combine their outputs.

        Split out from ``forward`` so subclasses can wrap the routed branch on
        its own -- e.g. running the experts inside a compressed latent space
        while routing and the shared branch stay at full width.

        Args:
            x (torch.Tensor): Expert inputs shaped ``[batch, seq_len, embed_dim]``.
            router_input (torch.Tensor, optional): Hidden states the router
                scores, shaped ``[batch, seq_len, router_embed_dim]``. Defaults
                to ``x``, i.e. routing in the same space as the experts.

        Returns:
            torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.
        """
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim) # (batch * seq, embed_dim)

        # Route to Experts
        top_k_indices, top_k_weights, router_logits = self.router(
            x if router_input is None else router_input
        ) # (batch, seq, top_k)
        self.last_router_logits = router_logits

        # Flatten for processing
        top_k_indices = top_k_indices.view(-1, self.top_k) # (batch * seq, top_k)
        top_k_weights = top_k_weights.view(-1, self.top_k) # (batch * seq, top_k)
        
        # Process with Experts
        # Current implementation: Loop over all experts (naive but correct)
        # For optimized implementations, we would group tokens by expert.
        
        final_output = torch.zeros_like(x_flat)
        
        # This is a naive implementation for correctness and simplicity.
        # Ideally, we should use scattered indices or sparse operations.
        # But given the 'PoorTorch' philosophy (efficiency < readability/educational), this is acceptable.
        
        # We process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert in any of the top-k positions
            # Create a mask: (batch * seq, top_k)
            expert_mask = (top_k_indices == i)
            
            # Any token that uses this expert?
            # (batch * seq)
            token_mask = expert_mask.any(dim=-1)
            
            if token_mask.any():
                # Select tokens for this expert
                expert_input = x_flat[token_mask]
                
                # Forward pass through expert
                expert_out = expert(expert_input) # (num_selected_tokens, embed_dim)
                
                # We need to add this output to the final accumulation, weighted by the router weight.
                # The token might have selected this expert at position k in top_k.
                
                # Get the weights for this expert for the selected tokens
                # We need to extract the specific weight corresponding to where 'i' was found in top_k_indices
                # Since an expert is selected at most once per token in top-k (usually), we can sum.
                
                # Create a weight vector matching expert_out
                # expert_mask[token_mask] selects the row of boolean flags for selected tokens
                # top_k_weights[token_mask] selects the weights for selected tokens
                
                # We mask the weights to only keep the one for this expert 
                # (batch * seq, top_k) * (batch * seq, top_k) -> sum over k -> (batch * seq)
                relevant_weights = (top_k_weights * expert_mask.float()).sum(dim=-1)
                selected_weights = relevant_weights[token_mask].unsqueeze(-1) # (num_tokens, 1)
                
                # Accumulate
                # We need to scatter add back to final_output
                # Instead of scatter, we can just index since we have boolean mask
                final_output[token_mask] += expert_out * selected_weights

        return final_output.view(batch_size, seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoE routing.

        Args:
            x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

        Returns:
            torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.
        """
        return self.compute_routed(x) + self.compute_shared(x)
