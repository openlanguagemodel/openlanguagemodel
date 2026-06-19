# `olm.models.facebook`

Source: [`src/olm/models/facebook/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/facebook/__init__.py#L1)

## Classes

### `OPT125M()`

**Bases:** `olm.models.facebook.opt.OPTModel`

Source: [`src/olm/models/facebook/opt.py:131`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/facebook/opt.py#L131)

OPT 125M Model Definition.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `OPTModel(vocab_size, embed_dim, intermediate_size, num_layers, num_heads, dropout=0.1, tie_weights=True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/facebook/opt.py:69`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/facebook/opt.py#L69)

OPT Model Definition.

Implements a decoder-only Transformer with specific OPT optimizations:
- Pre-normalization with LayerNorm
- Multi-Head Attention with Causal Masking
- ReLU activation in Feed-Forward Networks
- Tied output projection through ``OutputHead`` by default

Forward:
    Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
    ``[batch, seq_len, vocab_size]``.

Args:
    vocab_size (int): Vocabulary size.
    embed_dim (int): Embedding dimension.
    intermediate_size (int): FFN dimension.
    num_layers (int): Number of layers.
    num_heads (int): Number of heads.
    dropout (float, optional): Dropout probability. Defaults to 0.1.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
