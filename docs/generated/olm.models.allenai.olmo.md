# `olm.models.allenai.olmo`

Source: [`src/olm/models/allenai/olmo.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L1)

## Classes

### `OLMoBlock(embed_dim: int, intermediate_size: int, num_heads: int, max_seq_len: int, dropout: float)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/allenai/olmo.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L10)

A single Transformer block for the OLMo architecture.

Structure:
    x = x + Attn(LN(x))
    x = x + SwiGLU(LN(x))

Args:
    embed_dim (int): Model dimension.
    intermediate_size (int): FFN hidden dimension.
    num_heads (int): Number of attention heads.
    max_seq_len (int): Max context.
    dropout (float): Dropout probability.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `OLMoModel(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/allenai/olmo.py:68`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L68)

Base class for the OLMo (Open Language Model) architecture.

Structure:
    Embedding -> [OLMoBlock] x N -> LayerNorm -> tied OutputHead.

Forward:
    Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
    ``[batch, seq_len, vocab_size]``.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `OLMo_7B()`

**Bases:** `olm.models.allenai.olmo.OLMoModel`

Source: [`src/olm/models/allenai/olmo.py:113`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L113)

OLMo 7B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
