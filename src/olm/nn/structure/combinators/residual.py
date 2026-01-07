from olm.nn.structure.combinators.base import BaseCombinator
import torch.nn as nn
import torch

class Residual(BaseCombinator):
    """
    Residual wrapper that adds the block output to its input.

    Args:
        block: Module applied to the input before residual addition.

    Attributes:
        block: Module used for the residual transformation.
    """
    def __init__(self, block: nn.Module):
        """
        Initialize the residual combinator.

        Args:
            block: Module applied to the input before residual addition.
        """
        super().__init__()

        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block and add the result to the input.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection applied.
        """
        y = x + self.block(x)
        return y
