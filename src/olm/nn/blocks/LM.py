from olm.nn.structure.block import Block
from olm.nn.embeddings import Embedding
from olm.nn.blocks import TransformerBlock, OutputHead
from olm.nn.structure.combinators import Repeat

class LM(Block):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        dropout=0.0,
        causal=True,
        ff_multiplier=2.5,
    ):
        super().__init__([
            # Embedding
            Embedding(vocab_size, embed_dim),

            # Stack of transformer blocks
            Repeat(
                lambda: TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    causal=causal,
                    ff_multiplier=ff_multiplier,
                ),
                num_layers,
            ),

            # Final projection to logits
            OutputHead(embed_dim, vocab_size),
        ])
