from abc import ABC, abstractmethod
import torch
from typing import List


class TokenizerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert a list of token IDs back to text."""
        pass
