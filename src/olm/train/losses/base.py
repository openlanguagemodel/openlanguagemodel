from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class LossBase(nn.Module, ABC):
    """Base class for all loss modules."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply loss to ``logits`` and ``y``."""
        raise NotImplementedError
