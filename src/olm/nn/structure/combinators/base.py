import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseCombinator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass