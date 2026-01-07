# src/olm/nn/embeddings/positional/rope.py
import torch
import torch.nn as nn
from typing import Optional
from olm.nn.embeddings.positional.base import PositionalEmbeddingBase


class RotaryPositionalEmbedding(PositionalEmbeddingBase):
    """
    Rotary Positional Embedding (RoPE) as described in
    “RoFormer: Enhanced Transformer with Rotary Position Embedding” (arXiv 2104.09864).

    This module precomputes sin/cos rotation frequencies for a given head‐dim, and then applies to
    query/key representations via interleaving real/imag parts (or equivalently pairs of dims).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        """
        Args:
            head_dim: the dimension per attention head (must be even)
            base: the base for geometric progression of frequencies
            max_seq_len: maximum sequence length to support (for cache)
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (got {head_dim})")
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # precompute rotational frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # inv_freq shape: (head_dim/2,)
        # we store as buffer for caching
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cache sin/cos for positions up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # outer product: (max_seq_len, head_dim/2)
        freqs = torch.einsum(
            "i,j->ij", t, inv_freq
        )  # shape = [max_seq_len, head_dim/2]
        # now cos and sin
        emb_sin = freqs.sin()
        emb_cos = freqs.cos()
        # shape to broadcast: [max_seq_len, 1, head_dim/2]
        emb_sin = emb_sin.unsqueeze(1)
        emb_cos = emb_cos.unsqueeze(1)
        self.register_buffer("emb_sin", emb_sin, persistent=False)
        self.register_buffer("emb_cos", emb_cos, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor x.

        Args:
            x: shape (batch_size, seq_len, num_heads, head_dim)
            seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
                If None, assumes positions are 0..seq_len-1 for each batch.

        Returns:
            Tensor of same shape as x, with RoPE applied.
        """
        b, seq_len, n_heads, head_dim = x.shape
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"

        if seq_positions is None:
            # shape (seq_len,) then broadcast
            pos = (
                torch.arange(seq_len, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(b, seq_len)
            )
        else:
            pos = seq_positions

        # fetch sin/cos for these positions
        # emb_sin/cos shape: [max_seq_len, 1, head_dim/2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} greater than max_seq_len {self.max_seq_len}"
            )

        sin = self.emb_sin[pos]  # (batch, seq_len, 1, head_dim/2)
        cos = self.emb_cos[pos]  # same shape

        # apply to x — interleave halves of head_dim into pairs
        # split x into even/odd dims: (x_even, x_odd)
        x = x.view(b, seq_len, n_heads, head_dim // 2, 2)
        x_even = x[..., 0]  # (b, s, n_h, head_dim/2)
        x_odd = x[..., 1]

        # apply rotation: (x_even * cos − x_odd * sin, x_even * sin + x_odd * cos)
        # result shape: (b, s, n_h, head_dim/2, 2)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)  # shape (..., 2)
        x_rot = x_rot.view(b, seq_len, n_heads, head_dim)

        return x_rot


class PartialRotaryPositionalEmbedding(PositionalEmbeddingBase):
    """
    Partial Rotary Positional Embedding (LLaMA-style RoPE).

    Only applies rotary embeddings to a fraction of the head dimensions,
    leaving the remaining dimensions unchanged. This is the approach used
    in models like LLaMA, where typically 25-50% of dimensions are rotated.

    For example, with head_dim=128 and rotary_percentage=0.5, only the first
    64 dimensions are rotated, while the last 64 dimensions pass through unchanged.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percentage: float = 0.5,
        base: int = 10000,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            head_dim: the dimension per attention head
            rotary_percentage: fraction of head_dim to apply rotation to (0.0 to 1.0)
            base: the base for geometric progression of frequencies
            max_seq_len: maximum sequence length to support (for cache)
        """
        super().__init__()
        if not 0.0 < rotary_percentage <= 1.0:
            raise ValueError(
                f"rotary_percentage must be in (0.0, 1.0], got {rotary_percentage}"
            )

        self.head_dim = head_dim
        self.rotary_percentage = rotary_percentage
        self.base = base
        self.max_seq_len = max_seq_len

        # calculate rotary dimensions (must be even)
        rotary_dim = int(head_dim * rotary_percentage)
        if rotary_dim % 2 != 0:
            rotary_dim = rotary_dim - 1  # Make it even
        self.rotary_dim = rotary_dim

        if self.rotary_dim == 0:
            raise ValueError(
                f"rotary_dim is 0 with head_dim={head_dim} and "
                f"rotary_percentage={rotary_percentage}. Use a larger percentage."
            )

        # precompute rotational frequencies for the rotary dimensions only
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        # inv_freq shape: (rotary_dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cache sin/cos for positions up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # outer product: (max_seq_len, rotary_dim/2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # now cos and sin
        emb_sin = freqs.sin()
        emb_cos = freqs.cos()
        # shape to broadcast: [max_seq_len, 1, rotary_dim/2]
        emb_sin = emb_sin.unsqueeze(1)
        emb_cos = emb_cos.unsqueeze(1)
        self.register_buffer("emb_sin", emb_sin, persistent=False)
        self.register_buffer("emb_cos", emb_cos, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply partial rotary positional embedding to input tensor x.

        Args:
            x: shape (batch_size, seq_len, num_heads, head_dim)
            seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
                If None, assumes positions are 0..seq_len-1 for each batch.

        Returns:
            Tensor of same shape as x, with partial RoPE applied.
        """
        b, seq_len, n_heads, head_dim = x.shape
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"

        if seq_positions is None:
            # shape (seq_len,) then broadcast
            pos = (
                torch.arange(seq_len, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(b, seq_len)
            )
        else:
            pos = seq_positions

        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} greater than max_seq_len {self.max_seq_len}"
            )

        # split x into rotary and pass-through parts
        x_rot = x[..., : self.rotary_dim]  # (b, seq_len, n_heads, rotary_dim)
        x_pass = x[
            ..., self.rotary_dim :
        ]  # (b, seq_len, n_heads, head_dim - rotary_dim)

        # fetch sin/cos for these positions
        sin = self.emb_sin[pos]  # (batch, seq_len, 1, rotary_dim/2)
        cos = self.emb_cos[pos]  # same shape

        # apply rotation to the rotary part
        # split x_rot into even/odd dims: (x_even, x_odd)
        x_rot = x_rot.view(b, seq_len, n_heads, self.rotary_dim // 2, 2)
        x_even = x_rot[..., 0]  # (b, s, n_h, rotary_dim/2)
        x_odd = x_rot[..., 1]

        # apply rotation: (x_even * cos − x_odd * sin, x_even * sin + x_odd * cos)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)  # shape (..., 2)
        x_rot = x_rot.view(b, seq_len, n_heads, self.rotary_dim)

        # concatenate rotated and pass-through parts
        x_out = torch.cat([x_rot, x_pass], dim=-1)

        return x_out
