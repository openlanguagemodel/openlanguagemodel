# `olm.models.google.gemma2`

Source: [`src/olm/models/google/gemma2.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L1)

## Classes

### `Gemma2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, head_dim: int, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, query_pre_attn_scalar: float | None = 256.0)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/google/gemma2.py:39`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L39)

A single Transformer block for Gemma 2.

Implements the "Sandwich" Normalization pattern:
Norm -> Attn -> Norm -> Residual
Norm -> MLP  -> Norm -> Residual

#### Methods

##### `forward(self, x)`

Source: [`src/olm/models/google/gemma2.py:86`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L86)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Gemma2Embedding(vocab_size: int, embedding_dim: int)`

**Bases:** `olm.nn.embeddings.token_embed.Embedding`

Source: [`src/olm/models/google/gemma2.py:15`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L15)

Gemma 2 token embedding with hidden-size scaling.

#### Methods

##### `forward(self, x)`

Source: [`src/olm/models/google/gemma2.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L22)

Forward pass of the Embedding layer.

Args:
    x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token IDs.

Returns:
    torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).

### `Gemma2FinalLogitSoftcap(softcap: float | None = 30.0)`

**Bases:** `Module`

Source: [`src/olm/models/google/gemma2.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L26)

Gemma 2 final logit soft-capping.

#### Methods

##### `forward(self, logits)`

Source: [`src/olm/models/google/gemma2.py:33`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L33)

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

### `Gemma2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, final_logit_softcap: float | None = 30.0, query_pre_attn_scalar: float | None = 256.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/google/gemma2.py:108`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L108)

Base class for Gemma 2 models.

Structure:
    Scaled token embedding -> [Gemma2Block] x N -> RMSNorm ->
    tied OutputHead -> optional final logit softcap.

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

### `Gemma2_27B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:175`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L175)

Gemma 2 27B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Gemma2_2B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:209`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L209)

Gemma 2 2B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Gemma2_9B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:192`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L192)

Gemma 2 9B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
