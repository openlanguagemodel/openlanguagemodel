import torch.nn as nn
import torch
from typing import List, Union

class Block(nn.Module):
    """
    A sequential block similar to nn.Sequential but flexible.
    """
    def __init__(self, blocks: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    

