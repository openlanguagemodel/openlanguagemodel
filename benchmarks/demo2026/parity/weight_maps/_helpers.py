"""Shared machinery for explicit, shape-checked HF -> OLM weight mappings.

A :class:`WeightMap` is a list of :class:`MapEntry`. Each entry declares which
Hugging Face tensors it consumes and how to build the corresponding OLM tensor
from them. Because the build function is a pure tensor transform, the same
entry is reused for two purposes:

1. copying reference *weights* into the OLM model, and
2. transforming reference *gradients* into OLM parameter layout so gradient
   cosines compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn as nn


@dataclass
class MapEntry:
    """One OLM parameter built from one or more HF tensors."""

    olm_key: str
    hf_keys: Sequence[str]
    build: Callable[[Dict[str, torch.Tensor]], torch.Tensor]


class WeightMap:
    def __init__(self, entries: List[MapEntry]):
        self.entries = entries
        seen = set()
        for entry in entries:
            if entry.olm_key in seen:
                raise ValueError(f"Duplicate OLM key in weight map: {entry.olm_key}")
            seen.add(entry.olm_key)

    def olm_keys(self) -> List[str]:
        return [entry.olm_key for entry in self.entries]

    def hf_keys(self) -> List[str]:
        keys: List[str] = []
        for entry in self.entries:
            keys.extend(entry.hf_keys)
        return keys

    def check_completeness(
        self,
        olm_model: nn.Module,
        hf_model: nn.Module,
        hf_ignored: Sequence[str] = (),
    ) -> None:
        """Every OLM parameter must be produced exactly once; every HF
        parameter must be consumed (or explicitly ignored, e.g. aliases of
        tied weights)."""
        olm_params = {name for name, _ in olm_model.named_parameters()}
        mapped = set(self.olm_keys())
        missing = olm_params - mapped
        extra = mapped - olm_params
        if missing:
            raise ValueError(f"OLM parameters not covered by map: {sorted(missing)}")
        if extra:
            raise ValueError(f"Map produces unknown OLM parameters: {sorted(extra)}")

        hf_params = {name for name, _ in hf_model.named_parameters()}
        consumed = set(self.hf_keys()) | set(hf_ignored)
        unconsumed = hf_params - consumed
        if unconsumed:
            raise ValueError(f"HF parameters not consumed by map: {sorted(unconsumed)}")

        unknown = set(self.hf_keys()) - hf_params
        if unknown:
            raise ValueError(f"Map references unknown HF parameters: {sorted(unknown)}")

    def copy_weights(self, hf_model: nn.Module, olm_model: nn.Module) -> None:
        hf_state = dict(hf_model.named_parameters())
        olm_params = dict(olm_model.named_parameters())
        with torch.no_grad():
            for entry in self.entries:
                sources = {k: hf_state[k].detach() for k in entry.hf_keys}
                tensor = entry.build(sources)
                target = olm_params[entry.olm_key]
                if tensor.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch for {entry.olm_key}: built {tuple(tensor.shape)} "
                        f"vs target {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(dtype=target.dtype))

    def hf_gradient_for(self, olm_key: str, hf_model: nn.Module) -> torch.Tensor:
        """Return the HF gradient transformed into OLM parameter layout."""
        entry = next(e for e in self.entries if e.olm_key == olm_key)
        hf_params = dict(hf_model.named_parameters())
        grads = {}
        for key in entry.hf_keys:
            grad = hf_params[key].grad
            if grad is None:
                raise ValueError(f"HF parameter {key} has no gradient")
            grads[key] = grad.detach()
        return entry.build(grads)


def permute_rope_rows(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert a q/k projection from HF half-split RoPE layout to OLM's
    interleaved RoPE layout.

    Hugging Face Llama-style checkpoints order each head's rows as
    ``[pair-first-halves..., pair-second-halves...]`` (rotate_half). OLM's
    ``RotaryPositionalEmbedding`` rotates interleaved even/odd pairs, so rows
    must be re-interleaved: ``out[2i] = hf[i]``, ``out[2i+1] = hf[i + d/2]``.
    Both conventions use the same frequency for pair ``i``, so this is an
    exact re-parameterization.

    Works for 2-D weights ``(num_heads*head_dim, in)`` and 1-D biases.
    """
    out_dim = weight.shape[0]
    head_dim = out_dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE permutation")
    rest = weight.shape[1:]
    w = weight.reshape(num_heads, 2, head_dim // 2, *rest)
    w = w.transpose(1, 2)
    return w.reshape(out_dim, *rest)


def transpose_conv1d(weight: torch.Tensor) -> torch.Tensor:
    """HF GPT-2 Conv1D stores weights as (in, out); OLM Linear wants (out, in)."""
    return weight.t().contiguous()
