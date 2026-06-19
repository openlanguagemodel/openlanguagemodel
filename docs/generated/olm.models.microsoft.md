# `olm.models.microsoft`

Source: [`src/olm/models/microsoft/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/__init__.py#L1)

## Classes

### `Phi3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, activation: str = 'swiglu', dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/microsoft/phi3.py:83`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi3.py#L83)

Base class for Phi 3 models.

Structure:
    Embedding -> [Phi3Block] x N -> RMSNorm -> tied OutputHead.

Forward:
    Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
    ``[batch, seq_len, vocab_size]``.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Phi-3/Phi-3.5 official checkpoints use
    specialized LongRoPE/scaled RoPE behavior for long contexts, so exact
    long-context behavior may differ from the released Microsoft checkpoints.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Phi3_5_Mini()`

**Bases:** `olm.models.microsoft.phi3.Phi3Model`

Source: [`src/olm/models/microsoft/phi3.py:144`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi3.py#L144)

Phi-3.5 Mini 3.8B Model.

Uses the public checkpoint dimensions. LongRoPE factors are not represented
by this lightweight preset.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Phi3_Small()`

**Bases:** `olm.models.microsoft.phi3.Phi3Model`

Source: [`src/olm/models/microsoft/phi3.py:165`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi3.py#L165)

Phi-3 Small 7B Model.

Distinguished by GeGLU activations and the public checkpoint dimensions.
LongRoPE and Phi-3 Small's block-sparse/dense attention schedule are not
represented by this lightweight preset.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Phi4Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 250000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/microsoft/phi4.py:73`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L73)

Base class for Phi 4 models.

Structure:
    Embedding -> [Phi4Block] x N -> RMSNorm -> tied OutputHead.

Forward:
    Accepts token IDs shaped ``[batch, seq_len]`` and returns logits shaped
    ``[batch, seq_len, vocab_size]``.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Phi4_14B()`

**Bases:** `olm.models.microsoft.phi4.Phi4Model`

Source: [`src/olm/models/microsoft/phi4.py:130`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft/phi4.py#L130)

Phi-4 14B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
