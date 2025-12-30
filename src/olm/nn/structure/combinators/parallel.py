from olm.nn.structure.combinators.base import BaseCombinator
import torch
import torch.nn as nn
from typing import List, Callable, Union

class Parallel(BaseCombinator):
    def __init__(self, blocks: List[nn.Module], merge: Callable = None, dim: int = -1):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.merge = merge if merge is not None else (lambda x, d: torch.sum(torch.stack(x, dim=d), dim=d))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))

        return self.merge(outputs, self.dim)


