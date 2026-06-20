# `olm.models.meta.llama2`

Source: [`src/olm/models/meta/llama2.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L1)

## Classes

### `Llama2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama2.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L10)

A single Transformer block for Llama 2.

**Structure**

x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))

**Parameters**

- `embed_dim` (`int`): Model dimension.
- `intermediate_size` (`int`): FFN hidden dimension.
- `num_heads` (`int`): Number of attention heads.
- `num_kv_heads` (`int`): Number of KV heads. If == num_heads, uses MHA. If < num_heads, uses GQA.
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
