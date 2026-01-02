import torch
import torch.nn as nn
from typing import Optional
from olm.nn.attention.base import AttentionBase, AttentionwithRoPEBase


class MultiHeadAttention(AttentionBase):
    """
    Implements Multi-Head Attention (MHA) as described in "Attention Is All You Need".

    Splits the input into multiple heads, computes scaled dot-product attention for each,
    and concatenates the results. Supports causal masking for autoregressive models.

    Args:
        embed_dims (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
        causal (bool, optional): If True, applies a causal mask. Defaults to False.

    Attributes:
        scale (float): Scaling factor (1 / sqrt(head_dim)).
        causal (bool): Whether to apply a causal mask.
    """
    def __init__(self, embed_dims: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__(embed_dims, num_heads, dropout)
        self.causal = causal

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape [batch, heads, seq, head_dim].
            k (torch.Tensor): Key tensor of shape [batch, heads, seq, head_dim].
            v (torch.Tensor): Value tensor of shape [batch, heads, seq, head_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: The result of the attention mechanism applied to v.
        """
        # q, k, v: [batch, heads, seq, dim]
        # doing scaled dot product attention here
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        out = torch.matmul(attention_probs, v)
        return out

class MultiHeadAttentionwithRoPE(AttentionwithRoPEBase):
    """
    Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).

    Splits the input into multiple heads, computes scaled dot-product attention for each,
    and concatenates the results. Uses RoPE for positional information.

    Args:
        embed_dims (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        max_seq_len (int): Maximum sequence length.
        dropout (float, optional): Dropout probability on attention weights. Defaults to 0.0.
        causal (bool, optional): If True, applies a causal mask. Defaults to False.

    Attributes:
        scale (float): Scaling factor (1 / sqrt(head_dim)).
        causal (bool): Whether to apply a causal mask.
    """
    def __init__(self, embed_dims: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, bias: bool = True):
        super().__init__(embed_dims, num_heads, max_seq_len, dropout, bias=bias)
        self.causal = causal

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the scaled dot-product attention suited for RoPE.

        Args:
            q (torch.Tensor): Query tensor of shape [batch, heads, seq, head_dim].
            k (torch.Tensor): Key tensor of shape [batch, heads, seq, head_dim].
            v (torch.Tensor): Value tensor of shape [batch, heads, seq, head_dim].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: The result of the attention mechanism applied to v.
        """
        # q, k, v: [batch, heads, seq, dim]
        # doing scaled dot product attention here
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        out = torch.matmul(attention_probs, v)
        return out
