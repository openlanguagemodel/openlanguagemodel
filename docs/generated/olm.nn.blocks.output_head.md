# `olm.nn.blocks.output_head`

Source: [`src/olm/nn/blocks/output_head.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/output_head.py#L1)

## Classes

### `OutputHead(embed_dim: int, vocab_size: int, bias: bool = False, tied_embedding=None, tie_weights: bool = True, norm: torch.nn.modules.module.Module | None = None, use_norm: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/nn/blocks/output_head.py:56`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/output_head.py#L56)

Final normalization and vocabulary projection for language models.

``OutputHead`` applies a normalization layer and then maps hidden states to
vocabulary logits. The projection is tied to the input token embedding by
default. In the tied path, logits are computed as
``F.linear(hidden, embedding.weight)`` so the output head and token
embedding share one parameter matrix.

**Forward**

Accepts hidden states with shape ``[batch, seq_len, embed_dim]`` and
returns logits with shape ``[batch, seq_len, vocab_size]``.

**Parameters**

- `embed_dim` (`int`): The dimension of the embedding space.
- `vocab_size` (`int`): The size of the vocabulary.
- `bias` (`bool, optional`): Whether to include bias in the linear layer. Defaults to False.
- `tied_embedding` (`nn.Module | nn.Parameter, optional`): Embedding module or weight parameter to reuse for the output projection.
- `tie_weights` (`bool, optional`): Whether to reuse ``tied_embedding`` as the output projection matrix. Defaults to True.
- `norm` (`nn.Module, optional`): Normalization module before projection. Defaults to ``LayerNorm(embed_dim)``.
- `use_norm` (`bool, optional`): If False and ``norm`` is not provided, use an identity layer instead of LayerNorm. Defaults to True.

**Attributes**

- `blocks` (`nn.ModuleList`): ``[norm, projection]``.

#### Properties

- `projection` -> `Module`
  Projection module used after normalization.
- `weight` -> `Parameter`
  Output projection weight; tied to the embedding matrix by default.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.
