# `olm.models.meta`

Source: [`src/olm/models/meta/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/__init__.py#L1)

## Classes

### `Llama2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama2.py:80`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L80)

Base class for Llama 2 models.

**Structure**

Embedding -> [Llama2Block] x N -> RMSNorm -> tied OutputHead.

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

### `Llama2_13B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:149`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L149)

Llama 2 13B (MHA).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama2_70B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:165`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L165)

Llama 2 70B (GQA).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.

### `Llama2_7B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:133`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L133)

Llama 2 7B (MHA).

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
