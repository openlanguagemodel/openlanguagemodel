# `olm.models.microsoft.phi4`

Source: [`src/olm/models/microsoft/phi4.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L1)

## Classes

### `Phi4Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/microsoft/phi4.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L10)

A single Transformer block for Phi 4.

**Structure**

x = x + GQA(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))

**Parameters**

- `embed_dim` (`int`): Model dimension.
- `intermediate_size` (`int`): FFN hidden dimension.
- `num_heads` (`int`): Number of attention heads.
- `num_kv_heads` (`int`): Number of KV heads.
- `max_seq_len` (`int`): Max sequence length.
- `dropout` (`float`): Dropout probability.
- `rope_theta` (`float`): RoPE base.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Phi4Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 250000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/microsoft/phi4.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L73)

Base class for Phi 4 models.

**Structure**

Embedding -> [Phi4Block] x N -> RMSNorm -> tied OutputHead.

**Forward**

Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
``[batch, seq_len, vocab_size]``.

**Implementation Note**

This implementation uses standard Rotary Positional Embeddings (RoPE)
parameterized via `rope_theta`.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Phi4_14B()`

**Bases:** `olm.models.microsoft.phi4.Phi4Model`

Source: [`src/olm/models/microsoft/phi4.py:130`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L130)

Phi-4 14B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.
