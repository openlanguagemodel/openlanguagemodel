# `olm.models.alibaba.qwen2`

Source: [`src/olm/models/alibaba/qwen2.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L1)

## Classes

### `Qwen2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, rms_norm_eps: float = 1e-06)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/alibaba/qwen2.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L9)

A single Transformer block for Qwen 2.

Structure:
    x = x + GQA(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

Args:
    embed_dim (int): Model dimension.
    intermediate_size (int): FFN hidden dimension.
    num_heads (int): Number of attention heads.
    num_kv_heads (int): Number of KV heads.
    max_seq_len (int): Max sequence length.
    dropout (float): Dropout probability.
    rope_theta (float): RoPE base.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float, tie_weights: bool = True, dropout: float = 0.0, rms_norm_eps: float = 1e-06)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/alibaba/qwen2.py:44`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L44)

Base class for Qwen 2 / 2.5 models.

Structure:
    Embedding -> [Qwen2Block] x N -> RMSNorm -> tied OutputHead.

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

### `Qwen2_5_0_5B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:165`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L165)

Qwen 2.5 0.5B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_14B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:108`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L108)

Qwen 2.5 14B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_1_5B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:151`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L151)

Qwen 2.5 1.5B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_32B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:93`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L93)

Qwen 2.5 32B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_3B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:137`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L137)

Qwen 2.5 3B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_72B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:78`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L78)

Qwen 2.5 72B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Qwen2_5_7B()`

**Bases:** `olm.models.alibaba.qwen2.Qwen2Model`

Source: [`src/olm/models/alibaba/qwen2.py:123`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py#L123)

Qwen 2.5 7B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
