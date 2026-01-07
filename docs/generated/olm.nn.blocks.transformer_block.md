# olm.nn.blocks.transformer_block

### Classes

| [`TransformerBlock`](#olm.nn.blocks.transformer_block.TransformerBlock)(\*args, \*\*kwargs)   | A single Transformer block containing Multi-Head Attention and a FeedForward Network.   |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|

### *class* olm.nn.blocks.transformer_block.Block(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Lightweight sequential container for composable submodules.

Similar to `nn.Sequential`, but exposes the underlying list for
inspection or dynamic manipulation by higher-level builders.

* **Parameters:**
  **blocks** – Ordered list of modules applied to the input in sequence.

#### blocks

ModuleList storing the ordered blocks.

#### forward(x: torch.Tensor) → torch.Tensor

Apply each block to the input in sequence.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor after all blocks have been applied.

### *class* olm.nn.blocks.transformer_block.LayerNorm(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`NormBase`](olm.nn.norms.base.md#olm.nn.norms.base.NormBase)

Layer Normalization layer.

Implements Layer Normalization as described in “Layer Normalization” ([https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)).
Normalizes the input across the features dimension.

* **Parameters:**
  * **d_model** (*int*) – The dimension of the model to normalize.
  * **eps** (*float* *,* *optional*) – Small constant for numerical stability. Defaults to 1e-5.
  * **device** (*torch.device* *,* *optional*) – Target device.
  * **dtype** (*torch.dtype* *,* *optional*) – Target data type.

#### gamma

Learnable scale parameter.

* **Type:**
  nn.Parameter

#### beta

Learnable shift parameter.

* **Type:**
  nn.Parameter

#### forward(x: torch.Tensor) → torch.Tensor

Forward pass of LayerNorm.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch_size, sequence_length, d_model).
* **Returns:**
  Normalized output tensor of the same shape.
* **Return type:**
  torch.Tensor

### *class* olm.nn.blocks.transformer_block.MultiHeadAttentionwithRoPE(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`AttentionwithRoPEBase`](olm.nn.attention.base.md#olm.nn.attention.base.AttentionwithRoPEBase)

Implements Multi-Head Attention (MHA) with Rotary Positional Embedding (RoPE).

Splits the input into multiple heads, computes scaled dot-product attention for each,
and concatenates the results. Uses RoPE for positional information.

* **Parameters:**
  * **embed_dims** (*int*) – Total dimension of the model.
  * **num_heads** (*int*) – Number of parallel attention heads.
  * **max_seq_len** (*int*) – Maximum sequence length.
  * **dropout** (*float* *,* *optional*) – Dropout probability on attention weights. Defaults to 0.0.
  * **causal** (*bool* *,* *optional*) – If True, applies a causal mask. Defaults to False.

#### scale

Scaling factor (1 / sqrt(head_dim)).

* **Type:**
  float

#### causal

Whether to apply a causal mask.

* **Type:**
  bool

#### compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) → torch.Tensor

Computes the scaled dot-product attention suited for RoPE.

* **Parameters:**
  * **q** (*torch.Tensor*) – Query tensor of shape [batch, heads, seq, head_dim].
  * **k** (*torch.Tensor*) – Key tensor of shape [batch, heads, seq, head_dim].
  * **v** (*torch.Tensor*) – Value tensor of shape [batch, heads, seq, head_dim].
  * **mask** (*torch.Tensor* *,* *optional*) – Attention mask. Defaults to None.
* **Returns:**
  The result of the attention mechanism applied to v.
* **Return type:**
  torch.Tensor

### *class* olm.nn.blocks.transformer_block.Parallel(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Apply multiple blocks to the same input and merge their outputs.

The merge function takes a list of tensors and a dimension argument.

* **Parameters:**
  * **blocks** – Modules applied in parallel to the same input.
  * **merge** – Function that combines the list of outputs and a dimension.
  * **dim** – Dimension used by the merge function when applicable.

#### blocks

ModuleList storing the parallel blocks.

#### merge

Merge function used to combine outputs.

#### dim

Dimension passed to the merge function.

#### forward(x: torch.Tensor) → torch.Tensor

Apply all blocks in parallel and merge their outputs.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Merged output tensor.

### *class* olm.nn.blocks.transformer_block.Repeat(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Repeat a module a fixed number of times in sequence.

The module function should return a new module instance each call.

* **Parameters:**
  * **module_func** – Callable returning a new module instance.
  * **num_repeat** – Number of times to repeat the module.

#### module

Factory callable used to create new modules.

#### num_repeat

Number of repeats.

#### stack

ModuleList containing the repeated modules.

#### forward(x: torch.Tensor) → torch.Tensor

Apply the repeated modules in sequence.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor after all repeats.

### *class* olm.nn.blocks.transformer_block.Residual(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`BaseCombinator`](olm.nn.structure.combinators.base.md#olm.nn.structure.combinators.base.BaseCombinator)

Residual wrapper that adds the block output to its input.

* **Parameters:**
  **block** – Module applied to the input before residual addition.

#### block

Module used for the residual transformation.

#### forward(x: torch.Tensor) → torch.Tensor

Apply the block and add the result to the input.

* **Parameters:**
  **x** – Input tensor.
* **Returns:**
  Output tensor with residual connection applied.

### *class* olm.nn.blocks.transformer_block.SwiGLUFFN(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`FeedForwardBase`](olm.nn.feedforward.base.md#olm.nn.feedforward.base.FeedForwardBase)

SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).

This layer implements the gated linear unit with Swish (SiLU) activation, which has been
shown to improve performance over standard GELU/ReLU FFNs.

Structure:
: Input
  -> Linear(embed_dim -> 2 \* hidden_dim) [Splits into Gate and Value]
  -> SwiGLU(Gate \* SiLU(Value))
  -> Linear(hidden_dim -> embed_dim)
  -> Dropout

* **Parameters:**
  * **embed_dim** (*int*) – The dimension of the input and output.
  * **hidden_dim** (*int* *,* *optional*) – The intermediate inner dimension.
    If None, defaults to int(ff_multiplier \* embed_dim).
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **bias** (*bool* *,* *optional*) – Whether to use bias in linear layers. Defaults to True.
  * **ff_multiplier** (*float* *,* *optional*) – Multiplier for default hidden dimension. Defaults to 2.5 (commonly 8/3 for SwiGLU).

#### up_proj

Projects and splits input into gate and value parts.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### act

The activation function.

* **Type:**
  [SwiGLU](olm.nn.activations.md#olm.nn.activations.SwiGLU)

#### down_proj

Projects back to embedding dimension.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### dropout

Dropout layer.

* **Type:**
  nn.Dropout

#### forward(x)

Forward pass of the feedforward network.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, embed_dim).
* **Returns:**
  Output tensor of shape (batch, seq_len, embed_dim).
* **Return type:**
  torch.Tensor

### *class* olm.nn.blocks.transformer_block.TransformerBlock(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`Block`](olm.nn.structure.block.md#olm.nn.structure.block.Block)

A single Transformer block containing Multi-Head Attention and a FeedForward Network.

This block implements the standard Transformer architecture with pre-normalization,
Rotary Positional Embeddings (RoPE), and SwiGLU activation in the feedforward layer.
It supports causal masking for autoregressive modeling.

Structure:
: Input -> LayerNorm -> MHA(RoPE) -> Residual -> LayerNorm -> SwiGLU FFN -> Residual -> Output

* **Parameters:**
  * **embed_dim** (*int*) – The dimension of the embedding space (d_model).
  * **num_heads** (*int*) – Number of attention heads. verify that embed_dim % num_heads == 0.
  * **max_seq_len** (*int*) – Maximum sequence length supported by the model (for RoPE).
  * **dropout** (*float* *,* *optional*) – Dropout probability for attention and FFN. Defaults to 0.0.
  * **causal** (*bool* *,* *optional*) – Whether to apply causal masking in attention. Defaults to False.
  * **ff_multiplier** (*float* *,* *optional*) – Multiplier for the hidden dimension of the FFN.
    Commonly 4.0 (standard) or 8/3 (SwiGLU). Defaults to 2.5.

#### layers

The sequential list of layers within the block.

* **Type:**
  nn.ModuleList
