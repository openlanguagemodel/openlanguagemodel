from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class LMOutput:
    """
    Structured language-model output for trainers that need auxiliary signals.

    ``logits`` is the only required field. Existing models may keep returning a
    plain tensor; the Trainer normalizes both forms internally.
    """

    logits: torch.Tensor
    aux_losses: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | None = None
    mtp_logits: list[torch.Tensor] | None = None
    router_stats: Any = None
    metadata: Mapping[str, Any] | None = None


def as_lm_output(output: torch.Tensor | LMOutput | Mapping[str, Any]) -> LMOutput:
    """Normalize tensor, dict, or ``LMOutput`` model returns."""
    if isinstance(output, LMOutput):
        return output

    if torch.is_tensor(output):
        return LMOutput(logits=output)

    if isinstance(output, Mapping):
        if "logits" not in output:
            raise ValueError("Structured model output dictionaries must include logits")
        return LMOutput(
            logits=output["logits"],
            aux_losses=output.get("aux_losses"),
            mtp_logits=output.get("mtp_logits"),
            router_stats=output.get("router_stats"),
            metadata=output.get("metadata"),
        )

    raise TypeError(
        "Model output must be a logits tensor, LMOutput, or mapping with logits"
    )
