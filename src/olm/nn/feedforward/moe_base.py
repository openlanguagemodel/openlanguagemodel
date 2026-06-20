from typing import Optional, List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

from olm.nn.feedforward.base import FeedForwardBase
from olm.nn.torch_nn_wrappers import Linear

class MoERouter(nn.Module):
    """
    Router for Mixture of Experts.
    
    Routes input tokens to the top-k experts based on learned gate logits.
    """
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = Linear(embed_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

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
        
        return top_k_indices, top_k_weights

class MoEFeedForwardBase(FeedForwardBase):
    """
    Base class for Mixture of Experts FeedForward networks.
    
    Supports:
    - Top-K routing
    - Shared experts (always active)
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
            **kwargs: Additional arguments passed to FeedForwardBase.
        """
        super().__init__(embed_dim)
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_kwargs = expert_kwargs or {}
        
        # Initialize Router
        self.router = MoERouter(embed_dim, num_experts, top_k)
        
        # Initialize Routable Experts
        self.experts = nn.ModuleList([
            expert_cls(embed_dim, **self.expert_kwargs) 
            for _ in range(num_experts)
        ])
        
        # Initialize Shared Experts (if any)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                expert_cls(embed_dim, **self.expert_kwargs)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoE routing.

        Args:
            x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

        Returns:
            torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.
        """
        identity = x
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim) # (batch * seq, embed_dim)
        
        # 1. Compute Shared Experts Output
        shared_output = 0
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x)
        
        # 2. Route to Experts
        top_k_indices, top_k_weights = self.router(x) # (batch, seq, top_k)
        
        # Flatten for processing
        top_k_indices = top_k_indices.view(-1, self.top_k) # (batch * seq, top_k)
        top_k_weights = top_k_weights.view(-1, self.top_k) # (batch * seq, top_k)
        
        # 3. Process with Experts
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

        final_output = final_output.view(batch_size, seq_len, embed_dim)
        
        # 4. Combine
        if self.shared_experts is not None:
            final_output = final_output + shared_output
            
        return final_output
