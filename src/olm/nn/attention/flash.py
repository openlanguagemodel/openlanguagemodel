# src/olm/nn/attention/flash.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from olm.nn.attention.base import AttentionBase, AttentionwithRoPEBase
from olm.nn.embeddings.positional.rope import RotaryPositionalEmbedding
import warnings


class FlashAttention(AttentionBase):
    """
    Flash Attention implementation for efficient attention computation.

    Uses PyTorch's native scaled_dot_product_attention (which includes Flash Attention 2
    optimizations) when available, or falls back to a memory-efficient implementation.

    Flash Attention provides:
    - O(N) memory complexity instead of O(N²) for sequence length N
    - Faster computation through kernel fusion and tiling
    - Exact attention (not an approximation)
    - Support for causal masking without materializing the full attention matrix

    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (Dao et al., 2022) and "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning" (Dao, 2023)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights (default: 0.0)
        causal: If True, applies causal masking for autoregressive models (default: False)
        use_flash_attn: Force enable/disable flash attention. If None, auto-detect (default: None)

    Example:
        >>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
        >>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 128, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash_attn: Optional[bool] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.causal = causal

        # Check if PyTorch scaled_dot_product_attention is available (PyTorch >= 2.0)
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

        if use_flash_attn is None:
            # Auto-detect: use flash attention if available
            self.use_flash_attn = self.has_sdpa
        else:
            self.use_flash_attn = use_flash_attn
            if use_flash_attn and not self.has_sdpa:
                warnings.warn(
                    "Flash Attention requested but scaled_dot_product_attention not available. "
                    "Requires PyTorch >= 2.0. Falling back to standard attention."
                )
                self.use_flash_attn = False

        # Store dropout probability for SDPA
        self.dropout_p = dropout

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes attention using Flash Attention when available.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]
            v: Value tensor [batch, heads, seq, head_dim]
            mask: Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]

        Returns:
            Attention output [batch, heads, seq, head_dim]
        """
        if self.use_flash_attn:
            return self._flash_attention(q, k, v, mask)
        else:
            return self._standard_attention(q, k, v, mask)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flash Attention implementation using PyTorch's scaled_dot_product_attention.

        This automatically uses Flash Attention 2 kernels when available and falls back
        to memory-efficient attention or math attention based on hardware and input shapes.
        """
        # PyTorch's SDPA handles causal masking efficiently
        is_causal = self.causal and mask is None

        # Convert custom mask if provided
        # SDPA expects mask with 0 for positions to attend and -inf for masked positions
        attn_mask = None
        if mask is not None:
            # If mask has 1s for valid positions and 0s for masked, convert it
            if mask.dtype == torch.bool:
                attn_mask = mask
            else:
                # Assume mask has 1s for valid, 0s for masked
                attn_mask = mask.bool()

        # Use PyTorch's optimized scaled_dot_product_attention
        # This will automatically select the best kernel (Flash Attention 2, memory efficient, or math)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        return out

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention (fallback).

        Uses the classic O(N²) memory implementation when Flash Attention is not available.
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal and mask is None:
            seq_len = q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply custom mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, D = x.shape

        # Project to Q, K, V and reshape to [batch, heads, seq, head_dim]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        out = self.compute_attention(q, k, v, mask)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        """String representation of the module."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, "
            f"flash_attn={self.use_flash_attn}"
        )


class FlashAttentionwithRoPE(AttentionwithRoPEBase):
    """
    Flash Attention implementation for efficient attention computation.

    Uses PyTorch's native scaled_dot_product_attention (which includes Flash Attention 2
    optimizations) when available, or falls back to a memory-efficient implementation.

    Flash Attention provides:
    - O(N) memory complexity instead of O(N²) for sequence length N
    - Faster computation through kernel fusion and tiling
    - Exact attention (not an approximation)
    - Support for causal masking without materializing the full attention matrix

    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (Dao et al., 2022) and "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning" (Dao, 2023)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights (default: 0.0)
        causal: If True, applies causal masking for autoregressive models (default: False)
        use_flash_attn: Force enable/disable flash attention. If None, auto-detect (default: None)

    Example:
        >>> attn = FlashAttention(embed_dim=512, num_heads=8, causal=True)
        >>> x = torch.randn(2, 128, 512)  # (batch, seq_len, embed_dim)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 128, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash_attn: Optional[bool] = None,
    ):
        super().__init__(embed_dim, num_heads, max_seq_len, dropout)
        self.causal = causal

        # Check if PyTorch scaled_dot_product_attention is available (PyTorch >= 2.0)
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")

        if use_flash_attn is None:
            # Auto-detect: use flash attention if available
            self.use_flash_attn = self.has_sdpa
        else:
            self.use_flash_attn = use_flash_attn
            if use_flash_attn and not self.has_sdpa:
                warnings.warn(
                    "Flash Attention requested but scaled_dot_product_attention not available. "
                    "Requires PyTorch >= 2.0. Falling back to standard attention."
                )
                self.use_flash_attn = False

        # Store dropout probability for SDPA
        self.dropout_p = dropout

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes attention using Flash Attention when available.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]
            v: Value tensor [batch, heads, seq, head_dim]
            mask: Optional attention mask [batch, heads, seq, seq] or [batch, 1, seq, seq]

        Returns:
            Attention output [batch, heads, seq, head_dim]
        """
        if self.use_flash_attn:
            return self._flash_attention(q, k, v, mask)
        else:
            return self._standard_attention(q, k, v, mask)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flash Attention implementation using PyTorch's scaled_dot_product_attention.

        This automatically uses Flash Attention 2 kernels when available and falls back
        to memory-efficient attention or math attention based on hardware and input shapes.
        """
        # PyTorch's SDPA handles causal masking efficiently
        is_causal = self.causal and mask is None

        # Convert custom mask if provided
        # SDPA expects mask with 0 for positions to attend and -inf for masked positions
        attn_mask = None
        if mask is not None:
            # If mask has 1s for valid positions and 0s for masked, convert it
            if mask.dtype == torch.bool:
                attn_mask = mask
            else:
                # Assume mask has 1s for valid, 0s for masked
                attn_mask = mask.bool()

        # Use PyTorch's optimized scaled_dot_product_attention
        # This will automatically select the best kernel (Flash Attention 2, memory efficient, or math)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        return out

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention (fallback).

        Uses the classic O(N²) memory implementation when Flash Attention is not available.
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal and mask is None:
            seq_len = q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply custom mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention and RoPE.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, D = x.shape

        # Project to Q, K, V and reshape to [batch, seq, heads, head_dim]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K (RoPE expects [batch, seq, heads, head_dim])
        q = self.rope(q)
        k = self.rope(k)

        # Transpose to [batch, heads, seq, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        out = self.compute_attention(q, k, v, mask)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        """String representation of the module."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, "
            f"flash_attn={self.use_flash_attn}"
        )
