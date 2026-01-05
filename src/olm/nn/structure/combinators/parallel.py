from olm.nn.structure.combinators.base import BaseCombinator
import torch
import torch.nn as nn
from typing import List, Callable, Union

class Parallel(BaseCombinator):
    """
    Apply multiple blocks to the same input and merge their outputs.

    The merge function takes a list of tensors and a dimension argument.

    Args:
        blocks: Modules applied in parallel to the same input.
        merge: Function that combines the list of outputs and a dimension.
        dim: Dimension used by the merge function when applicable.

    Attributes:
        blocks: ModuleList storing the parallel blocks.
        merge: Merge function used to combine outputs.
        dim: Dimension passed to the merge function.
    """
    def __init__(self, blocks: List[nn.Module], merge: Callable = None, dim: int = -1):
        """
        Initialize the parallel combinator.

        Args:
            blocks: Modules applied in parallel to the same input.
            merge: Function that combines the list of outputs and a dimension.
            dim: Dimension used by the merge function when applicable.
        """
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.merge = merge if merge is not None else (lambda x, d: torch.sum(torch.stack(x, dim=d), dim=d))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all blocks in parallel and merge their outputs.

        Args:
            x: Input tensor.

        Returns:
            Merged output tensor.
        """
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))

        return self.merge(outputs, self.dim)
