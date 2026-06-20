# `olm.models.google`

Source: [`src/olm/models/google/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/__init__.py#L1)

## Classes

### `Gemma2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, final_logit_softcap: float | None = 30.0, query_pre_attn_scalar: float | None = 256.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/google/gemma2.py:108`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L108)

Base class for Gemma 2 models.

**Structure**

Scaled token embedding -> [Gemma2Block] x N -> RMSNorm ->
tied OutputHead -> optional final logit softcap.

**Forward**

Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
``[batch, seq_len, vocab_size]``.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Gemma2_27B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:175`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L175)

Gemma 2 27B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Gemma2_2B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:209`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L209)

Gemma 2 2B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Gemma2_9B()`

**Bases:** `olm.models.google.gemma2.Gemma2Model`

Source: [`src/olm/models/google/gemma2.py:192`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py#L192)

Gemma 2 9B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.
