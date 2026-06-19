# `olm.models.google.gemma2`

## Classes

### `Gemma2Block(embed_dim: int, intermediate_size: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float, rope_theta: float, head_dim: int, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, query_pre_attn_scalar: float | None = 256.0)`

A single Transformer block for Gemma 2.

Implements the "Sandwich" Normalization pattern:
Norm -> Attn -> Norm -> Residual
Norm -> MLP  -> Norm -> Residual

#### Methods

- `forward(self, x)`
  Apply each block to the input in sequence.

### `Gemma2Embedding(vocab_size: int, embedding_dim: int)`

Gemma 2 token embedding with hidden-size scaling.

#### Methods

- `forward(self, x)`
  Forward pass of the Embedding layer.

### `Gemma2FinalLogitSoftcap(softcap: float | None = 30.0)`

Gemma 2 final logit soft-capping.

#### Methods

- `forward(self, logits)`
  Define the computation performed at every call.

### `Gemma2Model(vocab_size: int, embed_dim: int, intermediate_size: int, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, dropout: float = 0.0, sliding_window: int | None = 4096, attn_logit_softcap: float | None = 50.0, final_logit_softcap: float | None = 30.0, query_pre_attn_scalar: float | None = 256.0, tie_weights: bool = True)`

Base class for Gemma 2 models.

### `Gemma2_27B()`

Gemma 2 27B Model.

### `Gemma2_2B()`

Gemma 2 2B Model.

### `Gemma2_9B()`

Gemma 2 9B Model.
