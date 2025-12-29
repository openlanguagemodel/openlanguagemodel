
from block import Block
from combinators.parallel import Parallel
from combinators.repeat import Repeat
from src.nn import Encoder, Embedding, LayerNorm, LinearProjection,RoPe,MHA,Residual,SwiGLU,MLP,Dropout,SigmoidLinear,Dropout, Decoder


class Pipeline:
    """
    A sequential container for model blocks, similar to torch.nn.Sequential.

    This class represents the high-level structure of an OLM model, executing
    blocks in a defined order. It enforces that nested Pipelines are not allowed
    to keep the architecture flat and manageable.

    Attributes:
        blocks (list): A list of layers or blocks to execute sequentially.
    """
    def __init__(self, blocks):
        """
        Initializes the Pipeline with a sequence of blocks.

        Args:
            blocks (list): List of callable blocks (e.g., TransformerBlock, Linear).
        """
        self.blocks = blocks

    def forward(self, x):
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

            assert type(block) != Pipeline
            x = block.forward(x)

        return x
