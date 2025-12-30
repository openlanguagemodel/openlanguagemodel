from olm.nn.structure.block import Block
from olm.nn.structure.combinators import Repeat, Residual, Parallel
from olm.nn.attention.mha import MultiHeadAttentionwithRoPE
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import LayerNorm


class TransformerBlock(Block):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 max_seq_len,
                 dropout=0.0,
                 causal=False,
                 ff_multiplier = 2.5,  # or 2.66
    ):
        super().__init__([
            Block([
                ## MHA with RoPE
                Residual(
                    Block([
                        LayerNorm(embed_dim),
                        MultiHeadAttentionwithRoPE(embed_dim, num_heads, max_seq_len, dropout=dropout, causal=causal),
                    ]),
                ),

                ## Feedforward
                Residual(
                    Block([
                        LayerNorm(embed_dim),
                        SwiGLUFFN(embed_dim, hidden_dim=ff_multiplier*embed_dim, dropout=dropout, ff_multiplier=ff_multiplier),
                    ]),
                ),
            ]),
        ])
