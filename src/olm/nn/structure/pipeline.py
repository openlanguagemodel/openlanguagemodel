from .block import Block
import torch.nn as nn
import torch
from typing import List, Union

class Pipeline(nn.Module):
    """
    A sequential container for model blocks, similar to torch.nn.Sequential.

    This class represents the high-level structure of an OLM model, executing
    blocks in a defined order. It enforces that nested Pipelines are not allowed
    to keep the architecture flat and manageable.

    Attributes:
        blocks (nn.ModuleList): A list of layers or blocks to execute sequentially.
    """
    def __init__(self, blocks: List[nn.Module]):
        """
        Initializes the Pipeline with a sequence of blocks.

        Args:
            blocks (list): List of callable blocks (e.g., TransformerBlock, Linear).
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through each block in sequence.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output after passing through all blocks.

        Raises:
            AssertionError: If a block is an instance of Pipeline (nested pipelines forbidden).
        """
        for block in self.blocks:
            #pipeline is global only
            assert not isinstance(block, Pipeline), "Nested Pipelines are not allowed"
            x = block(x)

        return x
