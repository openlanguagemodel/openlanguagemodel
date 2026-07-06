import math
import torch
from typing import Optional

from olm.nn.embeddings.positional.base import PositionalEmbeddingBase


class Llama3RotaryPositionalEmbedding(PositionalEmbeddingBase):
    """
    Llama 3-style RoPE with frequency-aware position scaling.

    Unlike simple linear interpolation, Llama 3 RoPE applies different
    scaling to different frequency bands:
        - High-frequency components (wavelength < ``original_max_seq_len /
          low_freq_factor``) are NOT scaled -- they already resolve fine
          positional differences.
        - Low-frequency components (wavelength > ``original_max_seq_len /
          high_freq_factor``) are fully scaled by ``1/factor``.
        - Intermediate components are smoothly interpolated.

    This preserves short-range positional precision while extending
    long-range context.

    Used by Step 3.5 Flash (with per-layer theta), Llama 3.1/3.2.

    Args:
        head_dim: Per-head dimension (must be even).
        max_seq_len: Target (extended) maximum sequence length.
        base: Base theta for frequency computation.
        factor: Context extension factor.
        original_max_seq_len: Original pre-extension context length.
        low_freq_factor: Wavelength threshold below which no scaling is
            applied (default 1.0).
        high_freq_factor: Wavelength threshold above which full scaling
            is applied (default 32.0).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 262144,
        base: float = 10000.0,
        factor: float = 2.0,
        original_max_seq_len: int = 131072,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 32.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (got {head_dim})")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.factor = factor
        self.original_max_seq_len = original_max_seq_len
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("emb_sin", torch.empty(0, 1, head_dim // 2), persistent=False)
        self.register_buffer("emb_cos", torch.empty(0, 1, head_dim // 2), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq_base = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        low_freq_wavelen = self.original_max_seq_len / self.low_freq_factor
        high_freq_wavelen = self.original_max_seq_len / self.high_freq_factor

        wavelengths = 2 * math.pi / inv_freq_base

        inv_freq_scaled = inv_freq_base / self.factor

        smooth_factor = (self.original_max_seq_len / wavelengths - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth_factor = smooth_factor.clamp(0.0, 1.0)

        inv_freq = torch.where(
            wavelengths < high_freq_wavelen,
            inv_freq_base,
            torch.where(
                wavelengths > low_freq_wavelen,
                inv_freq_scaled,
                (1 - smooth_factor) * inv_freq_scaled + smooth_factor * inv_freq_base,
            ),
        )

        return inv_freq

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("emb_sin", freqs.sin().unsqueeze(1), persistent=False)
        self.register_buffer("emb_cos", freqs.cos().unsqueeze(1), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply Llama 3-style RoPE.

        Args:
            x: ``[batch, seq_len, num_heads, head_dim]``
            seq_positions: Optional position indices ``[batch, seq_len]``.

        Returns:
            Tensor with RoPE applied, same shape as input.
        """
        b, seq_len, n_heads, head_dim = x.shape
        assert head_dim == self.head_dim

        if seq_positions is None:
            pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(b, seq_len)
        else:
            pos = seq_positions

        required_len = int(pos.max().item()) + 1 if pos.numel() else seq_len
        if self.emb_sin.device != x.device or self.emb_sin.size(0) < required_len:
            self._update_cache(required_len, x.device)

        sin = self.emb_sin[pos].to(dtype=x.dtype)
        cos = self.emb_cos[pos].to(dtype=x.dtype)

        x = x.view(b, seq_len, n_heads, head_dim // 2, 2)
        x_even = x[..., 0]
        x_odd = x[..., 1]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
        return x_rot.view(b, seq_len, n_heads, head_dim)
