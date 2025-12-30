from olm.nn.structure.combinators.base import BaseCombinator
import torch.nn as nn
import torch

class Residual(BaseCombinator):
    def __init__(self, block: nn.Module):
        super().__init__()

        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.block(x)
        return y