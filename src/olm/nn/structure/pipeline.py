
from block import Block
from combinators.parallel import Parallel
from combinators.repeat import Repeat
from src.nn import Encoder, Embedding, LayerNorm, LinearProjection,RoPe,MHA,Residual,SwiGLU,MLP,Dropout,SigmoidLinear,Dropout, Decoder


class Pipeline:
    def __init__(self, blocks):
        self.structure = blocks

    def forward(self, x):
        for block in self.blocks:
            #pipeline is global only

            assert type(block) != Pipeline
            x = block.forward(x)

        return x
