# `olm.models`

Source: [`src/olm/models/__init__.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/__init__.py#L1)

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

### `Llama2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama2.py:80`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L80)

Base class for Llama 2 models.

Structure:
    Embedding -> [Llama2Block] x N -> RMSNorm -> tied OutputHead.

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

### `Llama2_13B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:149`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L149)

Llama 2 13B (MHA).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama2_70B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:165`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L165)

Llama 2 70B (GQA).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama2_7B()`

**Bases:** `olm.models.meta.llama2.Llama2Model`

Source: [`src/olm/models/meta/llama2.py:133`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py#L133)

Llama 2 7B (MHA).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 500000.0, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/meta/llama3.py:75`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L75)

Base class for Llama 3, 3.1, and 3.2 models.

Inherits from Block for pure sequential composition.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Llama 3.1/3.2 official checkpoints use
    specialized scaled RoPE behavior for long contexts, so exact long-context
    behavior may differ from the released Meta checkpoints.

Structure:
    Embedding -> [Llama3Block] x N -> RMSNorm -> tied OutputHead.

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

### `Llama3_1_405B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:139`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L139)

Llama 3.1 405B Model (Flagship).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama3_1_70B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:155`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L155)

Llama 3.1 70B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama3_1_8B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:171`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L171)

Llama 3.1 8B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama3_2_1B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:206`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L206)

Llama 3.2 1B Model (Pruned/Distilled).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `Llama3_2_3B()`

**Bases:** `olm.models.meta.llama3.Llama3Model`

Source: [`src/olm/models/meta/llama3.py:190`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py#L190)

Llama 3.2 3B Model (Edge-optimized).

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

### `OLMoModel(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0, tie_weights: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/models/allenai/olmo.py:68`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L68)

Base class for the OLMo (Open Language Model) architecture.

Structure:
    Embedding -> [OLMoBlock] x N -> LayerNorm -> tied OutputHead.

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

### `OLMo_7B()`

**Bases:** `olm.models.allenai.olmo.OLMoModel`

Source: [`src/olm/models/allenai/olmo.py:113`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py#L113)

OLMo 7B Model.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.

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
