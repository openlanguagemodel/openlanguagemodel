"""Declarative weight-mapping primitives shared by every parity case.

A parity case never relies on parameter names lining up by coincidence. It
states, for every OLM parameter, exactly which HuggingFace parameter(s) it is
built from and the (linear) transform that builds it. The same declaration is
reused three ways:

1. to copy reference weights into the OLM model,
2. to prove that no parameter on either side was left unmapped,
3. to pair up gradients so gradient cosine similarity compares like with like.

Because every transform here is linear, applying it to a HuggingFace *gradient*
yields the gradient that the corresponding OLM parameter should have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch


# --------------------------------------------------------------------------
# Mapping records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MapEntry:
    """One OLM parameter expressed in terms of HuggingFace parameters.

    Args:
        olm: Fully qualified OLM parameter name (as in ``named_parameters``).
        hf: One HF parameter name, or a tuple of names when the OLM parameter
            fuses several reference tensors (e.g. a single ``up_proj`` holding
            both halves of a gated MLP).
        fn: Linear transform mapping the HF tensor(s) to the OLM layout.
            Defaults to identity / plain concatenation-free passthrough.
        note: Human-readable reason, surfaced in the README and in failures.
    """

    olm: str
    hf: str | tuple[str, ...]
    fn: Callable[..., torch.Tensor] | None = None
    note: str = ""
    #: Jacobian of ``fn``, used when mapping gradients. Only needed when ``fn``
    #: is affine rather than linear: for ``w -> w + 1`` the weight transform
    #: shifts, but ``dL/d(olm_w) == dL/d(hf_w)``, so the gradient transform is
    #: the identity. Defaults to ``fn``, which is correct for every linear map.
    grad_fn: Callable[..., torch.Tensor] | None = None

    @property
    def hf_names(self) -> tuple[str, ...]:
        return (self.hf,) if isinstance(self.hf, str) else tuple(self.hf)

    def _call(self, fn, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if fn is None:
            if len(tensors) != 1:
                raise ValueError(
                    f"{self.olm}: multiple HF sources require an explicit fn"
                )
            return tensors[0]
        return fn(*tensors)

    def apply(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Map reference weights into the OLM layout."""
        return self._call(self.fn, tensors)

    def apply_grad(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Map reference gradients into the OLM layout."""
        return self._call(self.grad_fn if self.grad_fn is not None else self.fn, tensors)


@dataclass
class ParityCase:
    """A single architecture to check against its HuggingFace reference."""

    name: str
    reference: str
    build_hf: Callable[[], torch.nn.Module]
    build_olm: Callable[[], torch.nn.Module]
    build_map: Callable[[torch.nn.Module, torch.nn.Module], Sequence[MapEntry]]
    #: Parameters that legitimately exist on only one side, with a reason.
    olm_only: dict[str, str] = field(default_factory=dict)
    hf_only: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    #: Set when OLM is known to differ from the stock reference config, and the
    #: reference must be nudged to isolate the rest of the graph. Recorded and
    #: reported rather than silently applied.
    reference_deviations: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def identity(t: torch.Tensor) -> torch.Tensor:
    return t


def transpose(t: torch.Tensor) -> torch.Tensor:
    """HF ``Conv1D`` stores ``[in, out]``; ``nn.Linear`` wants ``[out, in]``."""
    return t.t().contiguous()


def cat_rows(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate along the output-feature axis."""
    return torch.cat(tensors, dim=0)


def rows(start: int, stop: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Slice an output-feature range out of a fused reference projection."""

    def _slice(t: torch.Tensor) -> torch.Tensor:
        return t[start:stop].contiguous()

    return _slice


def add_scalar(value: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gemma-style ``(1 + w)`` norm scale folded into a plain ``w`` scale."""

    def _add(t: torch.Tensor) -> torch.Tensor:
        return t + value

    return _add


def compose(*fns: Callable[[torch.Tensor], torch.Tensor]):
    """Right-to-left composition, so ``compose(f, g)(x) == f(g(x))``."""

    def _composed(t: torch.Tensor) -> torch.Tensor:
        for fn in reversed(fns):
            t = fn(t)
        return t

    return _composed


# --------------------------------------------------------------------------
# RoPE convention bridge
# --------------------------------------------------------------------------


def interleave_permutation(head_dim: int) -> torch.Tensor:
    """Index permutation taking half-split RoPE layout to interleaved layout.

    HuggingFace's Llama-family ``rotate_half`` pairs dimension ``i`` with
    ``i + head_dim/2``. OLM's :class:`RotaryPositionalEmbedding` pairs adjacent
    dimensions ``2j`` and ``2j+1``. The two are the *same* rotation viewed
    through a permutation of the head axis: OLM slot ``2j`` corresponds to HF
    slot ``j`` and OLM slot ``2j+1`` to HF slot ``j + head_dim/2``.

    This is the same permutation Meta's official checkpoint-conversion script
    applies to ``q_proj``/``k_proj``, in the opposite direction.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


def rope_permute(num_heads: int, head_dim: int):
    """Reorder the head axis of a Q/K projection into OLM's RoPE convention.

    Applied to ``q_proj`` and ``k_proj`` only. ``v_proj`` and the output
    projection are deliberately left alone: attention logits are invariant
    under the *same* permutation applied to both Q and K, so values and the
    output projection stay in the reference layout.
    """
    perm = interleave_permutation(head_dim)

    def _permute(t: torch.Tensor) -> torch.Tensor:
        shape = t.shape
        if shape[0] != num_heads * head_dim:
            raise ValueError(
                f"expected leading dim {num_heads * head_dim}, got {shape[0]}"
            )
        view = t.reshape(num_heads, head_dim, *shape[1:])
        return view[:, perm].reshape(shape).contiguous()

    return _permute
