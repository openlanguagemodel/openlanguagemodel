# `olm.models.meta.llama3`

Source: [`src/olm/models/meta/llama3.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L1)

## Classes

### `Llama3Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama3.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L10)

A single Transformer block for Llama 3.x architecture.

Similar to Llama 2 but parameterized for Llama 3's high-performance context.

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

### `Llama3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 500000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama3.py:75`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L75)

Base class for Llama 3, 3.1, and 3.2 models.

Inherits from Block for pure sequential composition.

**Implementation Note**

This implementation uses standard Rotary Positional Embeddings (RoPE)
parameterized via `rope_theta`. Llama 3.1/3.2 official checkpoints use
specialized scaled RoPE behavior for long contexts, so exact long-context
behavior may differ from the released Meta checkpoints.

**Structure**

Embedding -> [Llama3Block] x N -> RMSNorm -> tied OutputHead.

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

### `Llama3_1_405B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:139`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L139)

Llama 3.1 405B Model (Flagship).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama3_1_70B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:155`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L155)

Llama 3.1 70B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama3_1_8B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:171`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L171)

Llama 3.1 8B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama3_2_1B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:206`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L206)

Llama 3.2 1B Model (Pruned/Distilled).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama3_2_3B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:190`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L190)

Llama 3.2 3B Model (Edge-optimized).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.
