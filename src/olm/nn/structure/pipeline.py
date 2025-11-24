
from block import Block
from combinators.parallel import Parallel
from combinators.repeat import Repeat
from src.nn import Encoder, Embedding, LayerNorm, LinearProjection,RoPe,MHA,Residual,SwiGLU,MLP,Dropout,SigmoidLinear,Dropout, Decoder

Block([
    Encoder(),
    Embedding(),
    Repeat(lambda: Block(
            Residual(
                Block([
                    LayerNorm(),
                    LinearProjection(),
                    RoPe(),
                    MHA(),
                ])
            ),
            LayerNorm(),
            SwiGLU(),
            MLP(),
            Dropout()
        ), 32
    ),
    LayerNorm(),
    MLP(),
    SigmoidLinear(),
    Decoder()
])


