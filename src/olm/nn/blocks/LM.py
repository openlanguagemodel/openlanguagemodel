import torch
from olm.nn.structure import Block
from olm.nn.embeddings import Embedding
from .transformer_block import TransformerBlock
from .output_head import OutputHead
from olm.nn.structure.combinators import Repeat


class LM(Block):
    """
    GPT-style causal language model assembled from OLM blocks.

    ``LM`` is the small, configurable model used throughout the beginner
    examples. It consists of a token embedding, ``num_layers`` repeated
    ``TransformerBlock`` modules, and an ``OutputHead`` that projects hidden
    states back to vocabulary logits. The output projection reuses the input
    embedding matrix by default.

    Structure:
        ``input_ids`` -> ``Embedding`` -> ``TransformerBlock`` x N ->
        ``OutputHead`` -> logits.

    Forward:
        Accepts integer token IDs with shape ``[batch, seq_len]`` and returns
        logits with shape ``[batch, seq_len, vocab_size]``. The inherited
        ``Block.forward`` applies each submodule sequentially.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embeddings and hidden states.
        num_heads (int): Number of attention heads in Transformer blocks.
        num_layers (int): Number of Transformer blocks.
        max_seq_len (int): Maximum sequence length for the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal masking. Defaults to True.
        ff_multiplier (float, optional): Multiplier for FFN hidden dimension. Defaults to 2.5.
        tie_embeddings (bool, optional): Whether the output head should reuse
            the input embedding matrix. Defaults to True.

    Attributes:
        blocks (nn.ModuleList): ``[embedding, transformer_stack, output_head]``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = True,
        ff_multiplier: float = 2.5,
        tie_embeddings: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        super().__init__(
            [
                # Embedding
                embedding,
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
                OutputHead(
                    embed_dim,
                    vocab_size,
                    tied_embedding=embedding,
                    tie_weights=tie_embeddings,
                ),
            ]
        )
