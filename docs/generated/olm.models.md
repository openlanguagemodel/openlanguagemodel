# `olm.models`

## Classes

### `GPT2()`

GPT-2 Small (124M).

### `GPT2Large()`

GPT-2 Large (774M).

### `GPT2Medium()`

GPT-2 Medium (355M).

### `GPT2Model(vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1)`

Base class for GPT-2 models.

### `GPT2XL()`

GPT-2 XL (1.5B).

### `Gemma2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, final_logit_softcap: float | None = 30.0, query_pre_attn_scalar: float | None = 256.0)`

Base class for Gemma 2 models.

### `Gemma2_27B()`

Gemma 2 27B Model.

### `Gemma2_2B()`

Gemma 2 2B Model.

### `Gemma2_9B()`

Gemma 2 9B Model.

### `Llama2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0)`

Base class for Llama 2 models.

Structure:
    Embedding -> [Llama2Block] x N -> RMSNorm -> Linear Head

### `Llama2_13B()`

Llama 2 13B (MHA).

### `Llama2_70B()`

Llama 2 70B (GQA).

### `Llama2_7B()`

Llama 2 7B (MHA).

### `Llama3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 500000.0, dropout: float = 0.0, tie_weights: bool = False)`

Base class for Llama 3, 3.1, and 3.2 models.

Inherits from Block for pure sequential composition.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Llama 3.1/3.2 official checkpoints use
    specialized scaled RoPE behavior for long contexts, so exact long-context
    behavior may differ from the released Meta checkpoints.

Structure:
    Embedding -> [Llama3Block] x N -> RMSNorm -> Linear Head

### `Llama3_1_405B()`

Llama 3.1 405B Model (Flagship).

### `Llama3_1_70B()`

Llama 3.1 70B Model.

### `Llama3_1_8B()`

Llama 3.1 8B Model.

### `Llama3_2_1B()`

Llama 3.2 1B Model (Pruned/Distilled).

### `Llama3_2_3B()`

Llama 3.2 3B Model (Edge-optimized).

### `OLMoModel(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0, tie_weights: bool = False)`

Base class for the OLMo (Open Language Model) architecture.

### `OLMo_7B()`

OLMo 7B Model.

### `OPT125M()`

OPT 125M Model Definition.

### `OPTModel(vocab_size, embed_dim, intermediate_size, num_layers, num_heads, dropout=0.1)`

OPT Model Definition.

Implements a decoder-only Transformer with specific OPT optimizations:
- Pre-normalization with LayerNorm
- Multi-Head Attention with Causal Masking
- ReLU activation in Feed-Forward Networks

Args:
    vocab_size (int): Vocabulary size.
    embed_dim (int): Embedding dimension.
    intermediate_size (int): FFN dimension.
    num_layers (int): Number of layers.
    num_heads (int): Number of heads.
    dropout (float, optional): Dropout probability. Defaults to 0.1.

### `Phi3Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 10000.0, activation: str = 'swiglu', dropout: float = 0.0, tie_weights: bool = False)`

Base class for Phi 3 models.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`. Phi-3/Phi-3.5 official checkpoints use
    specialized LongRoPE/scaled RoPE behavior for long contexts, so exact
    long-context behavior may differ from the released Microsoft checkpoints.

### `Phi3_5_Mini()`

Phi-3.5 Mini 3.8B Model.

Uses the public checkpoint dimensions. LongRoPE factors are not represented
by this lightweight preset.

### `Phi3_Small()`

Phi-3 Small 7B Model.

Distinguished by GeGLU activations and the public checkpoint dimensions.
LongRoPE and Phi-3 Small's block-sparse/dense attention schedule are not
represented by this lightweight preset.

### `Phi4Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float = 250000.0, dropout: float = 0.0, tie_weights: bool = False)`

Base class for Phi 4 models.

Implementation note:
    This implementation uses standard Rotary Positional Embeddings (RoPE)
    parameterized via `rope_theta`.

### `Phi4_14B()`

Phi-4 14B Model.

### `Qwen2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, rope_theta: float, tie_weights: bool = False, dropout: float = 0.0, rms_norm_eps: float = 1e-06)`

Base class for Qwen 2 / 2.5 models.

Structure:
    Embedding -> [Qwen2Block] x N -> RMSNorm -> Linear Head

### `Qwen2_5_0_5B()`

Qwen 2.5 0.5B Model.

### `Qwen2_5_14B()`

Qwen 2.5 14B Model.

### `Qwen2_5_1_5B()`

Qwen 2.5 1.5B Model.

### `Qwen2_5_32B()`

Qwen 2.5 32B Model.

### `Qwen2_5_3B()`

Qwen 2.5 3B Model.

### `Qwen2_5_72B()`

Qwen 2.5 72B Model.

### `Qwen2_5_7B()`

Qwen 2.5 7B Model.
