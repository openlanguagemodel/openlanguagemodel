# `olm.nn.blocks.output_head`

## Classes

### `OutputHead(embed_dim: int, vocab_size: int, bias: bool = False)`

Final output projection layer for the Language Model.

Consists of a LayerNorm followed by a Linear projection to the vocabulary size.
Typical structure: LayerNorm -> Linear(vocab_size).

Args:
    embed_dim (int): The dimension of the embedding space.
    vocab_size (int): The size of the vocabulary.
    bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.

Attributes:
    layers (nn.ModuleList): The normalization and linear layers.
