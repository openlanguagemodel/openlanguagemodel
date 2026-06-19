# Neural Network Components API

Composable PyTorch modules for language-model architectures.

## Modules

| Module | Public API |
|---|---|
| [`olm.nn.activations.base`](../generated/olm.nn.activations.base.md) | `ActivationBase` |
| [`olm.nn.activations.elu`](../generated/olm.nn.activations.elu.md) | `ELU` |
| [`olm.nn.activations.geglu`](../generated/olm.nn.activations.geglu.md) | `GeGLU` |
| [`olm.nn.activations.gelu`](../generated/olm.nn.activations.gelu.md) | `GELU` |
| [`olm.nn.activations.glu`](../generated/olm.nn.activations.glu.md) | `GLU` |
| [`olm.nn.activations.identity`](../generated/olm.nn.activations.identity.md) | `Identity` |
| [`olm.nn.activations.leaky_relu`](../generated/olm.nn.activations.leaky_relu.md) | `LeakyReLU` |
| [`olm.nn.activations.liglu`](../generated/olm.nn.activations.liglu.md) | `LiGLU` |
| [`olm.nn.activations.mish`](../generated/olm.nn.activations.mish.md) | `Mish` |
| [`olm.nn.activations.prelu`](../generated/olm.nn.activations.prelu.md) | `PReLU` |
| [`olm.nn.activations.reglu`](../generated/olm.nn.activations.reglu.md) | `ReGLU` |
| [`olm.nn.activations.relu`](../generated/olm.nn.activations.relu.md) | `ReLU` |
| [`olm.nn.activations.selu`](../generated/olm.nn.activations.selu.md) | `SELU` |
| [`olm.nn.activations.sigmoid`](../generated/olm.nn.activations.sigmoid.md) | `Sigmoid` |
| [`olm.nn.activations.silu`](../generated/olm.nn.activations.silu.md) | `SiLU`, `Swish` |
| [`olm.nn.activations.softmax`](../generated/olm.nn.activations.softmax.md) | `Softmax` |
| [`olm.nn.activations.softplus`](../generated/olm.nn.activations.softplus.md) | `Softplus` |
| [`olm.nn.activations.swiglu`](../generated/olm.nn.activations.swiglu.md) | `SwiGLU` |
| [`olm.nn.activations.swish`](../generated/olm.nn.activations.swish.md) | `Swish` |
| [`olm.nn.activations.tanh`](../generated/olm.nn.activations.tanh.md) | `Tanh` |
| [`olm.nn.attention`](../generated/olm.nn.attention.md) | `AttentionBase`, `AttentionwithRoPEBase`, `FlashAttention`, `FlashAttentionwithRoPE`, `GroupedQueryAttention`, `MultiHeadAttention`, `MultiHeadAttentionwithALiBi`, `MultiHeadAttentionwithRoPE` |
| [`olm.nn.attention.alibi`](../generated/olm.nn.attention.alibi.md) | `MultiHeadAttentionwithALiBi` |
| [`olm.nn.attention.base`](../generated/olm.nn.attention.base.md) | `AttentionBase`, `AttentionwithRoPEBase` |
| [`olm.nn.attention.flash`](../generated/olm.nn.attention.flash.md) | `FlashAttention`, `FlashAttentionwithRoPE` |
| [`olm.nn.attention.gqa`](../generated/olm.nn.attention.gqa.md) | `GroupedQueryAttention` |
| [`olm.nn.attention.masks`](../generated/olm.nn.attention.masks.md) | `attention_mask_to_bool` |
| [`olm.nn.attention.mha`](../generated/olm.nn.attention.mha.md) | `MultiHeadAttention`, `MultiHeadAttentionwithRoPE` |
| [`olm.nn.blocks.LM`](../generated/olm.nn.blocks.LM.md) | `LM` |
| [`olm.nn.blocks.linear_projections`](../generated/olm.nn.blocks.linear_projections.md) | `QKVProjection` |
| [`olm.nn.blocks.output_head`](../generated/olm.nn.blocks.output_head.md) | `OutputHead` |
| [`olm.nn.blocks.transformer_block`](../generated/olm.nn.blocks.transformer_block.md) | `TransformerBlock` |
| [`olm.nn.embeddings`](../generated/olm.nn.embeddings.md) | `AbsolutePositionalEmbedding`, `Embedding` |
| [`olm.nn.embeddings.positional`](../generated/olm.nn.embeddings.positional.md) | `ALiBiPositionalBias`, `AbsolutePositionalEmbedding`, `PartialRotaryPositionalEmbedding`, `PositionalEmbeddingBase`, `RotaryPositionalEmbedding`, `SinusoidalPositionalEmbedding` |
| [`olm.nn.embeddings.positional.absolute`](../generated/olm.nn.embeddings.positional.absolute.md) | `AbsolutePositionalEmbedding` |
| [`olm.nn.embeddings.positional.alibi`](../generated/olm.nn.embeddings.positional.alibi.md) | `ALiBiPositionalBias` |
| [`olm.nn.embeddings.positional.base`](../generated/olm.nn.embeddings.positional.base.md) | `PositionalEmbeddingBase` |
| [`olm.nn.embeddings.positional.rope`](../generated/olm.nn.embeddings.positional.rope.md) | `PartialRotaryPositionalEmbedding`, `PartialScaledRotaryPositionalEmbedding`, `RotaryPositionalEmbedding`, `ScaledRotaryPositionalEmbedding` |
| [`olm.nn.embeddings.positional.sinusoidal`](../generated/olm.nn.embeddings.positional.sinusoidal.md) | `SinusoidalPositionalEmbedding` |
| [`olm.nn.embeddings.token_embed`](../generated/olm.nn.embeddings.token_embed.md) | `Embedding` |
| [`olm.nn.feedforward`](../generated/olm.nn.feedforward.md) | `ClassicFFN`, `ClassicMoEFFN`, `FeedForwardBase`, `GeGLUFFN`, `GeGLUMoEFFN`, `SwiGLUFFN`, `SwiGLUMoEFFN` |
| [`olm.nn.feedforward.base`](../generated/olm.nn.feedforward.base.md) | `FeedForwardBase` |
| [`olm.nn.feedforward.classic_ffn`](../generated/olm.nn.feedforward.classic_ffn.md) | `ClassicFFN` |
| [`olm.nn.feedforward.classic_moe`](../generated/olm.nn.feedforward.classic_moe.md) | `ClassicMoEFFN` |
| [`olm.nn.feedforward.geglu_ffn`](../generated/olm.nn.feedforward.geglu_ffn.md) | `GeGLUFFN` |
| [`olm.nn.feedforward.geglu_moe`](../generated/olm.nn.feedforward.geglu_moe.md) | `GeGLUMoEFFN` |
| [`olm.nn.feedforward.moe_base`](../generated/olm.nn.feedforward.moe_base.md) | `MoEFeedForwardBase`, `MoERouter` |
| [`olm.nn.feedforward.swiglu_ffn`](../generated/olm.nn.feedforward.swiglu_ffn.md) | `SwiGLUFFN` |
| [`olm.nn.feedforward.swiglu_moe`](../generated/olm.nn.feedforward.swiglu_moe.md) | `SwiGLUMoEFFN` |
| [`olm.nn.norms`](../generated/olm.nn.norms.md) | `LayerNorm`, `RMSNorm` |
| [`olm.nn.norms.base`](../generated/olm.nn.norms.base.md) | `NormBase` |
| [`olm.nn.norms.layer_norm`](../generated/olm.nn.norms.layer_norm.md) | `LayerNorm` |
| [`olm.nn.norms.rms_norm`](../generated/olm.nn.norms.rms_norm.md) | `RMSNorm` |
| [`olm.nn.structure.block`](../generated/olm.nn.structure.block.md) | `Block`, `load`, `load_block`, `load_model` |
| [`olm.nn.structure.combinators`](../generated/olm.nn.structure.combinators.md) | `BaseCombinator`, `Parallel`, `Repeat`, `Residual` |
| [`olm.nn.structure.combinators.base`](../generated/olm.nn.structure.combinators.base.md) | `BaseCombinator` |
| [`olm.nn.structure.combinators.parallel`](../generated/olm.nn.structure.combinators.parallel.md) | `Parallel` |
| [`olm.nn.structure.combinators.repeat`](../generated/olm.nn.structure.combinators.repeat.md) | `Repeat` |
| [`olm.nn.structure.combinators.residual`](../generated/olm.nn.structure.combinators.residual.md) | `Residual` |
| [`olm.nn.torch_nn_wrappers`](../generated/olm.nn.torch_nn_wrappers.md) | `Linear` |
