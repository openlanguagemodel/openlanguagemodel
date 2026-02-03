# src/olm/nn/embeddings/positional/rope.py
import torch
import torch.nn as nn
from typing import Optional, Literal
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


class ScaledRotaryPositionalEmbedding(PositionalEmbeddingBase):
    """
    Scaled Rotary Positional Embedding with multiple scaling strategies.

    Supports the following scaling methods for extending context length:
    - 'linear': Linear position interpolation (Position Interpolation, arXiv:2306.15595)
    - 'ntk': NTK-aware scaling (dynamically adjusts base frequency)
    - 'dynamic_ntk': Dynamic NTK (adjusts base based on current sequence length)
    - 'yarn': YaRN (Yet another RoPE extensioN method, arXiv:2309.00071)
    - 'xpos': XPos (exponential decay for better extrapolation, arXiv:2212.10554)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_type: Literal[
            "linear", "ntk", "dynamic_ntk", "yarn", "xpos"
        ] = "linear",
        scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        # YaRN specific parameters
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
        # XPos specific parameters
        xpos_scale_base: Optional[int] = None,
    ):
        """
        Args:
            head_dim: the dimension per attention head (must be even)
            max_seq_len: maximum sequence length to support (for cache)
            base: the base for geometric progression of frequencies
            scaling_type: the type of scaling to apply
            scaling_factor: scaling factor for position (linear) or base adjustment (ntk)
            original_max_seq_len: original context length before scaling (for ntk/yarn)
            yarn_alpha: attention scale factor for YaRN
            yarn_beta: wavelength threshold for YaRN interpolation
            xpos_scale_base: base for XPos exponential scaling (defaults to max_seq_len)
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (got {head_dim})")

        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len or max_seq_len
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.xpos_scale_base = xpos_scale_base or max_seq_len

        # Compute effective base based on scaling type
        if scaling_type == "ntk":
            # NTK-aware scaling: adjust base frequency
            adjusted_base = base * (scaling_factor ** (head_dim / (head_dim - 2)))
            inv_freq = 1.0 / (
                adjusted_base
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        elif scaling_type == "yarn":
            # YaRN uses interpolation between low and high frequency dimensions
            inv_freq = self._compute_yarn_inv_freq()
        else:
            # Standard or linear/dynamic_ntk/xpos (computed differently)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Compute position embeddings
        self._update_embeddings(max_seq_len)

        # For XPos, compute scale factors
        if scaling_type == "xpos":
            scale = (
                torch.arange(0, head_dim, 2, dtype=torch.float32) + 0.4 * head_dim
            ) / (1.4 * head_dim)
            self.register_buffer("scale", scale, persistent=False)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for YaRN scaling."""
        dim_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)

        # Compute wavelengths
        inv_freq_base = 1.0 / (self.base ** (dim_indices / self.head_dim))
        wavelengths = 2 * torch.pi / inv_freq_base

        # Apply YaRN interpolation based on wavelength
        # Low frequency dims (long wavelengths) are not interpolated
        # High frequency dims (short wavelengths) are interpolated
        inv_freq_scaled = torch.where(
            wavelengths > self.yarn_beta,
            inv_freq_base,  # Keep low frequencies unchanged
            inv_freq_base / self.scaling_factor,  # Interpolate high frequencies
        )

        return inv_freq_scaled

    def _update_embeddings(self, seq_len: int):
        """Update sin/cos embeddings for given sequence length."""
        t = torch.arange(seq_len, dtype=torch.float32)

        if self.scaling_type == "linear":
            # Linear scaling: divide positions by scaling factor
            t = t / self.scaling_factor

        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Compute sin/cos
        emb_sin = freqs.sin().unsqueeze(1)
        emb_cos = freqs.cos().unsqueeze(1)

        self.register_buffer("emb_sin", emb_sin, persistent=False)
        self.register_buffer("emb_cos", emb_cos, persistent=False)

    def _get_dynamic_ntk_inv_freq(self, seq_len: int) -> torch.Tensor:
        """Compute inverse frequencies dynamically based on sequence length."""
        if seq_len <= self.original_max_seq_len:
            return self.inv_freq

        # Dynamic NTK: adjust base when sequence exceeds original max
        scale = seq_len / self.original_max_seq_len
        adjusted_base = self.base * (scale ** (self.head_dim / (self.head_dim - 2)))

        return 1.0 / (
            adjusted_base
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.inv_freq.device,
                )
                / self.head_dim
            )
        )

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply scaled rotary positional embedding to input tensor x.

        Args:
            x: shape (batch_size, seq_len, num_heads, head_dim)
            seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
                If None, assumes positions are 0..seq_len-1 for each batch.

        Returns:
            Tensor of same shape as x, with scaled RoPE applied.
        """
        b, seq_len, n_heads, head_dim = x.shape
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"

        if seq_positions is None:
            pos = (
                torch.arange(seq_len, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(b, seq_len)
            )
        else:
            pos = seq_positions

        # Handle dynamic NTK scaling
        if self.scaling_type == "dynamic_ntk" and seq_len > self.max_seq_len:
            # Recompute frequencies dynamically
            inv_freq = self._get_dynamic_ntk_inv_freq(seq_len)
            t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin = freqs.sin().unsqueeze(1)
            cos = freqs.cos().unsqueeze(1)
            sin = sin[pos]
            cos = cos[pos]
        else:
            # Use cached embeddings
            if seq_len > self.max_seq_len:
                # Extend cache if needed (for non-dynamic methods)
                self._update_embeddings(seq_len)
                self.max_seq_len = seq_len

            sin = self.emb_sin[pos]
            cos = self.emb_cos[pos]

        # Apply XPos scaling if enabled
        if self.scaling_type == "xpos":
            # Apply exponential decay based on position
            xpos_scale = self.scale ** pos.unsqueeze(-1).float()
            xpos_scale = xpos_scale.unsqueeze(-2)  # Add head dimension
            x = x * xpos_scale

        # Apply rotation
        x = x.view(b, seq_len, n_heads, head_dim // 2, 2)
        x_even = x[..., 0]
        x_odd = x[..., 1]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
        x_rot = x_rot.view(b, seq_len, n_heads, head_dim)

        return x_rot


class PartialScaledRotaryPositionalEmbedding(PositionalEmbeddingBase):
    """
    Partial Rotary Positional Embedding with scaling support.

    Combines partial RoPE (only rotating a fraction of dimensions) with
    various scaling strategies for extended context lengths.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percentage: float = 0.5,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_type: Literal[
            "linear", "ntk", "dynamic_ntk", "yarn", "xpos"
        ] = "linear",
        scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
        xpos_scale_base: Optional[int] = None,
    ):
        """
        Args:
            head_dim: the dimension per attention head
            rotary_percentage: fraction of head_dim to apply rotation to (0.0 to 1.0)
            max_seq_len: maximum sequence length to support
            base: the base for geometric progression of frequencies
            scaling_type: the type of scaling to apply
            scaling_factor: scaling factor for position or base adjustment
            original_max_seq_len: original context length before scaling
            yarn_alpha: attention scale factor for YaRN
            yarn_beta: wavelength threshold for YaRN
            xpos_scale_base: base for XPos exponential scaling
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
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len or max_seq_len
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.xpos_scale_base = xpos_scale_base or max_seq_len

        # Calculate rotary dimensions (must be even)
        rotary_dim = int(head_dim * rotary_percentage)
        if rotary_dim % 2 != 0:
            rotary_dim = rotary_dim - 1
        self.rotary_dim = rotary_dim

        if self.rotary_dim == 0:
            raise ValueError(
                f"rotary_dim is 0 with head_dim={head_dim} and rotary_percentage={rotary_percentage}"
            )

        # Compute inverse frequencies with scaling
        if scaling_type == "ntk":
            adjusted_base = base * (scaling_factor ** (rotary_dim / (rotary_dim - 2)))
            inv_freq = 1.0 / (
                adjusted_base
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
        elif scaling_type == "yarn":
            inv_freq = self._compute_yarn_inv_freq()
        else:
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_embeddings(max_seq_len)

        # XPos scale factors
        if scaling_type == "xpos":
            scale = (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32) + 0.4 * rotary_dim
            ) / (1.4 * rotary_dim)
            self.register_buffer("scale", scale, persistent=False)

    def _compute_yarn_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for YaRN scaling."""
        dim_indices = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        inv_freq_base = 1.0 / (self.base ** (dim_indices / self.rotary_dim))
        wavelengths = 2 * torch.pi / inv_freq_base

        inv_freq_scaled = torch.where(
            wavelengths > self.yarn_beta,
            inv_freq_base,
            inv_freq_base / self.scaling_factor,
        )

        return inv_freq_scaled

    def _update_embeddings(self, seq_len: int):
        """Update sin/cos embeddings for given sequence length."""
        t = torch.arange(seq_len, dtype=torch.float32)

        if self.scaling_type == "linear":
            t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb_sin = freqs.sin().unsqueeze(1)
        emb_cos = freqs.cos().unsqueeze(1)

        self.register_buffer("emb_sin", emb_sin, persistent=False)
        self.register_buffer("emb_cos", emb_cos, persistent=False)

    def _get_dynamic_ntk_inv_freq(self, seq_len: int) -> torch.Tensor:
        """Compute inverse frequencies dynamically based on sequence length."""
        if seq_len <= self.original_max_seq_len:
            return self.inv_freq

        scale = seq_len / self.original_max_seq_len
        adjusted_base = self.base * (scale ** (self.rotary_dim / (self.rotary_dim - 2)))

        return 1.0 / (
            adjusted_base
            ** (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=self.inv_freq.device,
                )
                / self.rotary_dim
            )
        )

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply partial scaled rotary positional embedding to input tensor x.

        Args:
            x: shape (batch_size, seq_len, num_heads, head_dim)
            seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.

        Returns:
            Tensor of same shape as x, with partial scaled RoPE applied.
        """
        b, seq_len, n_heads, head_dim = x.shape
        assert (
            head_dim == self.head_dim
        ), f"Expected head_dim={self.head_dim}, got {head_dim}"

        if seq_positions is None:
            pos = (
                torch.arange(seq_len, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(b, seq_len)
            )
        else:
            pos = seq_positions

        # Split into rotary and pass-through parts
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        # Handle dynamic NTK scaling
        if self.scaling_type == "dynamic_ntk" and seq_len > self.max_seq_len:
            inv_freq = self._get_dynamic_ntk_inv_freq(seq_len)
            t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin = freqs.sin().unsqueeze(1)[pos]
            cos = freqs.cos().unsqueeze(1)[pos]
        else:
            if seq_len > self.max_seq_len:
                self._update_embeddings(seq_len)
                self.max_seq_len = seq_len

            sin = self.emb_sin[pos]
            cos = self.emb_cos[pos]

        # Apply XPos scaling
        if self.scaling_type == "xpos":
            xpos_scale = self.scale ** pos.unsqueeze(-1).float()
            xpos_scale = xpos_scale.unsqueeze(-2)
            # Pad xpos_scale to match rotary dimensions
            xpos_scale_full = (
                xpos_scale.expand(-1, -1, n_heads, self.rotary_dim // 2)
                .reshape(b, seq_len, n_heads, self.rotary_dim // 2, 1)
                .expand(-1, -1, -1, -1, 2)
            )
            xpos_scale_full = xpos_scale_full.reshape(
                b, seq_len, n_heads, self.rotary_dim
            )
            x_rot = x_rot * xpos_scale_full

        # Apply rotation
        x_rot = x_rot.view(b, seq_len, n_heads, self.rotary_dim // 2, 2)
        x_even = x_rot[..., 0]
        x_odd = x_rot[..., 1]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
        x_rot = x_rot.view(b, seq_len, n_heads, self.rotary_dim)

        # Concatenate rotated and pass-through parts
        x_out = torch.cat([x_rot, x_pass], dim=-1)

        return x_out
