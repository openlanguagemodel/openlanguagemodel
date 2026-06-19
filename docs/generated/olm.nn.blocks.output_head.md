# `olm.nn.blocks.output_head`

## Classes

### `OutputHead(embed_dim: int, vocab_size: int, bias: bool = False, tied_embedding=None)`

Final output projection layer for the Language Model.

Consists of a LayerNorm followed by a projection to the vocabulary size.
By default the projection has its own weights. Pass ``tied_embedding`` to
share the projection matrix with the input token embedding.

Args:
    embed_dim (int): The dimension of the embedding space.
    vocab_size (int): The size of the vocabulary.
    bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.
    tied_embedding (nn.Module | nn.Parameter, optional): Embedding module or
        weight parameter to reuse for the output projection.

Attributes:
    layers (nn.ModuleList): The normalization and linear layers.
