# `olm.models.openai.gpt2`

Source: [`src/olm/models/openai/gpt2.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L1)

## Classes

### `GPT2()`

**Bases:** `olm.models.openai.gpt2.GPT2Model`

Source: [`src/olm/models/openai/gpt2.py:63`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L63)

GPT-2 Small (124M).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `GPT2Block(embed_dim: int, num_heads: int, dropout: float = 0.1)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/openai/gpt2.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L9)

A single Transformer block for GPT-2.

Structure:
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))

Args:
    embed_dim (int): Model dimension.
    num_heads (int): Number of attention heads.
    dropout (float): Dropout probability.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `GPT2Large()`

**Bases:** `olm.models.openai.gpt2.GPT2Model`

Source: [`src/olm/models/openai/gpt2.py:85`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L85)

GPT-2 Large (774M).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `GPT2Medium()`

**Bases:** `olm.models.openai.gpt2.GPT2Model`

Source: [`src/olm/models/openai/gpt2.py:74`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L74)

GPT-2 Medium (355M).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `GPT2Model(vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/openai/gpt2.py:34`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L34)

Base class for GPT-2 models.

Structure:
    Token embedding + learned positional embedding -> GPT2Block x N ->
    tied OutputHead.

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

### `GPT2XL()`

**Bases:** `olm.models.openai.gpt2.GPT2Model`

Source: [`src/olm/models/openai/gpt2.py:96`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py#L96)

GPT-2 XL (1.5B).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
