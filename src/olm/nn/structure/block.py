import torch.nn as nn
import torch
from typing import List, Union

class Block(nn.Module):
    """
    Lightweight sequential container for composable submodules.

    Similar to ``nn.Sequential``, but exposes the underlying list for
    inspection or dynamic manipulation by higher-level builders.

    Args:
        blocks: Ordered list of modules applied to the input in sequence.

    Attributes:
        blocks: ModuleList storing the ordered blocks.
    """
    def __init__(self, blocks: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply each block to the input in sequence.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all blocks have been applied.
        """
        for block in self.blocks:
            x = block(x)
        return x
    
