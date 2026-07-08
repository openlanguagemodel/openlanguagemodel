# src/olm/nn/attention/sparse.py
"""
Sparse attention with *real* sub-quadratic cost.

Unlike a mask-based "sparse" attention that still materializes the full
``N x N`` score matrix (and therefore stays O(N^2)), this implementation uses
PyTorch's FlexAttention (``torch.nn.attention.flex_attention``) together with a
pre-computed ``BlockMask``. FlexAttention inspects the block mask and *skips the
matmul entirely for blocks that are fully masked out*, so a sliding-window or
block-sparse pattern actually translates into less compute and less memory --
not just a different set of ``-inf`` entries.

Supported patterns (all causal-friendly for autoregressive LMs):

- ``"sliding_window"`` : each query attends to the previous ``window`` keys
  (local attention, à la Mistral / Longformer). Cost ~ O(N * window).
- ``"dilated"``        : a strided local band -- every ``dilation``-th key
  inside the window (à la LongNet / Sparse Transformer strided heads).
- ``"global"``         : sliding window **plus** the first ``global_tokens``
  keys are attended by everyone ("attention sink" / StreamingLLM-style global
  tokens). Keeps causality intact.

Backends:

- ``"flex"`` : FlexAttention + BlockMask. True block-sparsity, the fast path.
  Requires PyTorch >= 2.5. Best speed when ``compile=True`` (compiles the
  flex kernel with ``torch.compile``).
- ``"sdpa"`` : dense fallback. Builds the boolean keep-mask and routes through
  ``F.scaled_dot_product_attention``. Numerically identical, but O(N^2). Used
  automatically when FlexAttention is unavailable or when a per-batch padding
  ``mask`` is supplied (which FlexAttention cannot express through a static
  BlockMask).

References:
- "Generating Long Sequences with Sparse Transformers" (Child et al., 2019)
- "Longformer" (Beltagy et al., 2020), "Big Bird" (Zaheer et al., 2020)
- "Efficient Streaming LMs with Attention Sinks" (Xiao et al., 2023)
- PyTorch FlexAttention (Dong et al., 2024)
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.attention.base import AttentionBase, AttentionwithRoPEBase
from olm.nn.attention.masks import attention_mask_to_bool

# FlexAttention is optional (PyTorch >= 2.5). Import lazily-safe.
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _HAS_FLEX = True
except Exception:  # pragma: no cover - depends on torch version
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX = False


# Default FlexAttention block size. Sparsity is resolved at block granularity:
# a block is skipped only if *every* (q, kv) pair inside it is masked, so the
# effective window is rounded up to a multiple of this value.
_DEFAULT_BLOCK_SIZE = 128


def build_sparse_mask_mod(
    *,
    pattern: str = "sliding_window",
    window: int = 256,
    dilation: int = 1,
    global_tokens: int = 0,
    causal: bool = True,
) -> Callable[..., torch.Tensor]:
    """
    Build a ``mask_mod(b, h, q_idx, kv_idx) -> bool`` function.

    The returned callable is written purely with elementwise tensor ops, so the
    *same* function works two ways:

    - FlexAttention calls it with 0-d index tensors to build a ``BlockMask``.
    - The dense fallback calls it with broadcast ``[N, N]`` index grids to build
      a boolean keep-mask in one shot.

    This guarantees the flex and sdpa backends are bit-for-bit consistent.

    Args:
        pattern: one of ``"sliding_window"``, ``"dilated"``, ``"global"``.
        window: local band width (number of keys each query may look back on).
        dilation: stride within the band (``"dilated"`` pattern). 1 = dense band.
        global_tokens: number of leading keys that everyone attends to
            (attention-sink style). 0 disables global tokens.
        causal: enforce ``kv_idx <= q_idx``.

    Returns:
        A ``mask_mod`` callable returning a boolean tensor (True = keep).
    """
    if pattern not in ("sliding_window", "dilated", "global"):
        raise ValueError(
            f"Unknown sparse pattern {pattern!r}. Expected one of "
            "'sliding_window', 'dilated', 'global'."
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")

    use_dilation = dilation if pattern == "dilated" else 1
    use_global = global_tokens if pattern == "global" else 0

    def mask_mod(b, h, q_idx, kv_idx):
        # Signed distance; for causal LMs this is >= 0 on kept positions.
        delta = q_idx - kv_idx

        if causal:
            local = (kv_idx <= q_idx) & (delta < window)
        else:
            dist = torch.abs(delta)
            local = dist < window

        if use_dilation > 1:
            # Keep only every `dilation`-th key inside the band. The query's own
            # position (delta == 0) is always retained so a token can see itself.
            on_stride = (delta % use_dilation) == 0
            local = local & (on_stride | (delta == 0))

        keep = local

        if use_global > 0:
            # Leading keys act as always-visible global / sink tokens.
            sink = kv_idx < use_global
            keep = keep | sink
            if causal:
                # A sink key is only visible once it exists (kv <= q).
                keep = keep & (kv_idx <= q_idx)

        return keep

    return mask_mod


class _SparseAttentionMixin:
    """
    Shared sparse-attention machinery for both the plain and RoPE bases.

    Subclasses must already define ``self.scale``, ``self.num_heads``,
    ``self.head_dim`` and ``self.dropout`` (provided by ``AttentionBase`` /
    ``AttentionwithRoPEBase``). This mixin owns pattern config, the cached
    ``BlockMask`` objects, and the actual sparse/dense kernels.
    """

    def _init_sparse(
        self,
        *,
        causal: bool,
        pattern: str,
        window: int,
        dilation: int,
        global_tokens: int,
        block_size: int,
        backend: str,
        compile_flex: bool,
        dropout: float,
    ) -> None:
        if backend not in ("auto", "flex", "sdpa"):
            raise ValueError(
                f"backend must be 'auto', 'flex' or 'sdpa', got {backend!r}"
            )

        self.causal = causal
        self.pattern = pattern
        self.window = window
        self.dilation = dilation
        self.global_tokens = global_tokens
        self.block_size = block_size
        self.dropout_p = dropout

        resolved = "flex" if backend == "auto" else backend
        if resolved == "flex" and not _HAS_FLEX:
            resolved = "sdpa"
        self.backend = resolved

        self._mask_mod = build_sparse_mask_mod(
            pattern=pattern,
            window=window,
            dilation=dilation,
            global_tokens=global_tokens,
            causal=causal,
        )

        # Caches keyed by (seq_len, device-str). BlockMask / dense masks are
        # purely structural, so they can be reused across forward passes.
        self._block_mask_cache: dict = {}
        self._dense_mask_cache: dict = {}

        # Lazily-built compiled flex kernel (compiling is expensive, do it once).
        self._compile_flex = compile_flex and _HAS_FLEX
        self._flex_fn = None

    # -- FlexAttention (true sparse) path ---------------------------------
    def _get_block_mask(self, seq_len: int, device: torch.device):
        key = (seq_len, str(device))
        bm = self._block_mask_cache.get(key)
        if bm is None:
            # B=H=None -> broadcast the same structural mask over batch/heads.
            bm = create_block_mask(
                self._mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=self.block_size,
            )
            self._block_mask_cache[key] = bm
        return bm

    def _flex_kernel(self):
        if self._flex_fn is not None:
            return self._flex_fn
        if self._compile_flex:
            try:
                self._flex_fn = torch.compile(flex_attention)
            except Exception:
                self._flex_fn = flex_attention
        else:
            self._flex_fn = flex_attention
        return self._flex_fn

    def _flex_attention(self, q, k, v) -> torch.Tensor:
        """
        Sparse attention via FlexAttention and a precomputed BlockMask.

        FlexAttention inspects the block mask and skips the matmul for any block
        that is fully masked out, so the structured sparsity pattern translates
        into real compute and memory savings instead of a different set of -inf
        entries. This is the fast path (best with compile=True on CUDA).
        """
        seq_len = q.size(-2)
        block_mask = self._get_block_mask(seq_len, q.device)
        kernel = self._flex_kernel()
        return kernel(q, k, v, block_mask=block_mask, scale=self.scale)

    # -- Dense SDPA fallback ----------------------------------------------
    def _get_dense_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        m = self._dense_mask_cache.get(key)
        if m is None:
            idx = torch.arange(seq_len, device=device)
            q_idx = idx[:, None]  # [N, 1]
            kv_idx = idx[None, :]  # [1, N]
            m = self._mask_mod(None, None, q_idx, kv_idx).to(torch.bool)
            self._dense_mask_cache[key] = m
        return m

    def _dense_attention(self, q, k, v, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Dense fallback using PyTorch's scaled_dot_product_attention.

        Builds the boolean keep-mask for the sparsity pattern and applies it with
        the classic O(N^2) implementation. Numerically identical to the
        FlexAttention path but without the block-skipping speedup. Used when
        FlexAttention is unavailable or when a per-batch padding mask is supplied.
        """
        seq_len = q.size(-2)
        keep = self._get_dense_mask(seq_len, q.device)  # [N, N] bool

        if mask is not None:
            # Fold in a user-supplied padding / custom mask.
            keep = keep & attention_mask_to_bool(mask, device=q.device)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=keep,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
            is_causal=False,  # causality already baked into `keep`
        )
        return out

    # -- Dispatch ----------------------------------------------------------
    def _sparse_compute(self, q, k, v, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Routes the computation to the FlexAttention or dense backend.

        FlexAttention encodes only the static structural pattern, so a per-batch
        padding mask cannot ride along a shared BlockMask; whenever one is
        supplied the call falls back to the dense path (still correct, just
        O(N^2)).
        """
        if self.backend == "flex" and mask is None:
            return self._flex_attention(q, k, v)
        return self._dense_attention(q, k, v, mask)

    def _sparse_extra_repr(self) -> str:
        return (
            f"pattern={self.pattern}, window={self.window}, "
            f"dilation={self.dilation}, global_tokens={self.global_tokens}, "
            f"causal={self.causal}, backend={self.backend}, "
            f"block_size={self.block_size}"
        )


class SparseAttention(_SparseAttentionMixin, AttentionBase):
    """
    Sparse Attention implementation for efficient long-sequence attention.

    Restricts each query to a structured subset of keys (sliding window, dilated
    band, or window plus global "sink" tokens) and, on the FlexAttention backend,
    skips the fully-masked blocks of the attention matrix entirely instead of
    computing and discarding them.

    Sparse Attention provides:
    - Sub-quadratic compute and memory by skipping masked blocks (FlexAttention)
    - Structured patterns: sliding window, dilated band, and global/sink tokens
    - Exact attention within the chosen pattern (not a low-rank approximation)
    - A dense scaled_dot_product_attention fallback when FlexAttention is
      unavailable or a per-batch padding mask is supplied

    Reference: "Generating Long Sequences with Sparse Transformers" (Child et al.,
    2019), "Longformer" (Beltagy et al., 2020), "Big Bird" (Zaheer et al., 2020),
    "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023),
    and PyTorch FlexAttention (Dong et al., 2024)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights, applied in the dense
            fallback only; FlexAttention does not support attention-weight
            dropout (default: 0.0)
        causal: If True, applies causal masking for autoregressive models
            (default: True)
        pattern: Sparsity pattern, one of "sliding_window", "dilated", "global"
            (default: "sliding_window")
        window: Local band width; number of past keys each query attends to
            (default: 256)
        dilation: Stride within the band for the "dilated" pattern (default: 1)
        global_tokens: Number of leading sink tokens attended by everyone, used by
            the "global" pattern (default: 0)
        block_size: FlexAttention block granularity; sparsity is resolved per
            block so the effective window rounds up to a multiple of this
            (default: 128)
        backend: Compute backend: "auto" (flex if available else sdpa), "flex",
            or "sdpa" (default: "auto")
        compile: If True, compiles the FlexAttention kernel with torch.compile for
            maximum throughput; recommended on CUDA (default: False)
        bias: Whether to use bias in the linear projections (default: True)

    Example:
        >>> attn = SparseAttention(embed_dim=512, num_heads=8, causal=True,
        ...                        pattern="sliding_window", window=256)
        >>> x = torch.randn(2, 4096, 512)  # (batch, seq_len, embed_dim)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 4096, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        pattern: str = "sliding_window",
        window: int = 256,
        dilation: int = 1,
        global_tokens: int = 0,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        backend: str = "auto",
        compile: bool = False,
        bias: bool = True,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias=bias)
        self._init_sparse(
            causal=causal,
            pattern=pattern,
            window=window,
            dilation=dilation,
            global_tokens=global_tokens,
            block_size=block_size,
            backend=backend,
            compile_flex=compile,
            dropout=dropout,
        )

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes sparse attention using FlexAttention when available.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]
            v: Value tensor [batch, heads, seq, head_dim]
            mask: Optional attention mask [batch, heads, seq, seq] or
                [batch, 1, seq, seq]

        Returns:
            Attention output [batch, heads, seq, head_dim]
        """
        return self._sparse_compute(q, k, v, mask)

    def extra_repr(self) -> str:
        """String representation of the module."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, " + self._sparse_extra_repr()
        )


class SparseAttentionwithRoPE(_SparseAttentionMixin, AttentionwithRoPEBase):
    """
    Sparse Attention with Rotary Positional Embedding (RoPE).

    Restricts each query to a structured subset of keys (sliding window, dilated
    band, or window plus global "sink" tokens) and, on the FlexAttention backend,
    skips the fully-masked blocks of the attention matrix entirely instead of
    computing and discarding them. RoPE is applied to Q and K by the base forward
    before the sparse kernel runs. This is the variant intended for the OLM
    transformer block.

    Sparse Attention provides:
    - Sub-quadratic compute and memory by skipping masked blocks (FlexAttention)
    - Structured patterns: sliding window, dilated band, and global/sink tokens
    - Exact attention within the chosen pattern (not a low-rank approximation)
    - A dense scaled_dot_product_attention fallback when FlexAttention is
      unavailable or a per-batch padding mask is supplied

    Reference: "Generating Long Sequences with Sparse Transformers" (Child et al.,
    2019), "Longformer" (Beltagy et al., 2020), "Big Bird" (Zaheer et al., 2020),
    "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023),
    and PyTorch FlexAttention (Dong et al., 2024)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        max_seq_len: Maximum sequence length (for RoPE)
        dropout: Dropout probability on attention weights, applied in the dense
            fallback only; FlexAttention does not support attention-weight
            dropout (default: 0.0)
        causal: If True, applies causal masking for autoregressive models
            (default: True)
        pattern: Sparsity pattern, one of "sliding_window", "dilated", "global"
            (default: "sliding_window")
        window: Local band width; number of past keys each query attends to
            (default: 256)
        dilation: Stride within the band for the "dilated" pattern (default: 1)
        global_tokens: Number of leading sink tokens attended by everyone, used by
            the "global" pattern (default: 0)
        block_size: FlexAttention block granularity; sparsity is resolved per
            block so the effective window rounds up to a multiple of this
            (default: 128)
        backend: Compute backend: "auto" (flex if available else sdpa), "flex",
            or "sdpa" (default: "auto")
        compile: If True, compiles the FlexAttention kernel with torch.compile for
            maximum throughput; recommended on CUDA (default: False)
        bias: Whether to use bias in the linear projections (default: True)
        rope_theta: Base frequency for RoPE (default: 10000.0)

    Example:
        >>> attn = SparseAttentionwithRoPE(embed_dim=512, num_heads=8,
        ...                                max_seq_len=4096, causal=True,
        ...                                pattern="sliding_window", window=256)
        >>> x = torch.randn(2, 4096, 512)  # (batch, seq_len, embed_dim)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 4096, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = True,
        pattern: str = "sliding_window",
        window: int = 256,
        dilation: int = 1,
        global_tokens: int = 0,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        backend: str = "auto",
        compile: bool = False,
        bias: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            max_seq_len,
            dropout,
            bias=bias,
            rope_theta=rope_theta,
        )
        self._init_sparse(
            causal=causal,
            pattern=pattern,
            window=window,
            dilation=dilation,
            global_tokens=global_tokens,
            block_size=block_size,
            backend=backend,
            compile_flex=compile,
            dropout=dropout,
        )

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes sparse attention using FlexAttention when available.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]
            v: Value tensor [batch, heads, seq, head_dim]
            mask: Optional attention mask [batch, heads, seq, seq] or
                [batch, 1, seq, seq]

        Returns:
            Attention output [batch, heads, seq, head_dim]
        """
        return self._sparse_compute(q, k, v, mask)

    def extra_repr(self) -> str:
        """String representation of the module."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            + self._sparse_extra_repr()
        )
