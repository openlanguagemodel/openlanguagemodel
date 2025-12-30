from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class LossBase(nn.Module, ABC):
    """Base class for all loss modules."""

    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def _apply_reduction(self, values, mask=None):
        if self.reduction == "none":
            return values

        if mask is None:
            if self.reduction == "sum":
                return values.sum()
            return values.mean()

        values = values * mask
        denom = mask.sum().clamp_min(1)

        if self.reduction == "sum":
            return values.sum()
        return values.sum() / denom


    @abstractmethod
    def forward(self, logits: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply loss to ``logits`` and ``y``."""
        raise NotImplementedError
