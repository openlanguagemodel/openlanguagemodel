# `olm.nn.blocks.output_head`

## Classes

### `OutputHead(embed_dim: int, vocab_size: int, bias: bool = False, tied_embedding=None, tie_weights: bool = True, norm: torch.nn.modules.module.Module | None = None, use_norm: bool = True)`

Final output projection layer for the Language Model.

Consists of a LayerNorm followed by a projection to the vocabulary size.
The projection is tied to the input token embedding by default; pass
``tie_weights=False`` when you want a separate output matrix.

Args:
    embed_dim (int): The dimension of the embedding space.
    vocab_size (int): The size of the vocabulary.
    bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.
    tied_embedding (nn.Module | nn.Parameter, optional): Embedding module or
        weight parameter to reuse for the output projection.
    tie_weights (bool, optional): Whether to reuse ``tied_embedding`` as
        the output projection matrix. Defaults to True.
    norm (nn.Module, optional): Normalization module before projection.
        Defaults to ``LayerNorm(embed_dim)``.
    use_norm (bool, optional): If False and ``norm`` is not provided, use
        an identity layer instead of LayerNorm. Defaults to True.

Attributes:
    layers (nn.ModuleList): The normalization and linear layers.
