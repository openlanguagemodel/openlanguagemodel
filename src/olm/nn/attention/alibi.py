import torch
import torch.nn as nn
from typing import Optional
from olm.nn.attention.base import AttentionBase
from olm.nn.embeddings.positional.alibi import ALiBiPositionalBias

class MultiHeadAttentionwithALiBi(AttentionBase):
    """
    Multi-Head Attention with ALiBi (Attention with Linear Biases).

    ALiBi adds a static, non-learned bias to attention scores based on the distance between
    query and key positions. This allows the model to extrapolate to longer sequence lengths
    than seen during training.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to False.
        causal (bool, optional): Whether to apply causal masking logic. Defaults to True.
        max_seq_len (int, optional): Max sequence length for precomputing ALiBi bias. Defaults to 2048.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False, causal: bool = True, max_seq_len: int = 2048):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias)
        self.causal = causal
        
        # Use existing ALiBi implementation
        self.alibi = ALiBiPositionalBias(num_heads, max_seq_len=max_seq_len)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes attention scores with ALiBi bias.
        """
        B, H, N, D = q.shape
        # q, k: [B, Heads, N, D]
        
        # Standard attention scores: Q @ K.T / sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Get ALiBi bias
        # Returns: [1, H, N, N]
        alibi_bias = self.alibi(N, N, device=q.device)
        
        # Add bias to logits
        attn_scores = attn_scores + alibi_bias

        # Apply Causal Mask if needed
        # ALiBi biases favor nearby tokens, but causal masking is still strictly required to prevent looking ahead.
        if mask is not None:
             attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        elif self.causal:
             # Create causal mask
             causal_mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
             attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
             
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        return out
