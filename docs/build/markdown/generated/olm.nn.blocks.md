# olm.nn.blocks

### *class* olm.nn.blocks.LM(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`Block`](olm.nn.structure.block.md#olm.nn.structure.block.Block)

A simple Language Model (LM) architecture.

This model consists of an embedding layer, a stack of Transformer blocks,
and a final output projection to the vocabulary size. It is designed for
causal language modeling (next-token prediction).

Structure:
: Input IDs -> Embedding -> [TransformerBlock] x N -> OutputHead -> Logits

* **Parameters:**
  * **vocab_size** (*int*) – Size of the vocabulary.
  * **embed_dim** (*int*) – Dimension of the embeddings and hidden states.
  * **num_heads** (*int*) – Number of attention heads in Transformer blocks.
  * **num_layers** (*int*) – Number of Transformer blocks.
  * **max_seq_len** (*int*) – Maximum sequence length for the model.
  * **dropout** (*float* *,* *optional*) – Dropout probability. Defaults to 0.0.
  * **causal** (*bool* *,* *optional*) – Whether to use causal masking. Defaults to True.
  * **ff_multiplier** (*float* *,* *optional*) – Multiplier for FFN hidden dimension. Defaults to 2.5.

#### layers

The sequence of layers in the model.

* **Type:**
  nn.ModuleList

### *class* olm.nn.blocks.OutputHead(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: [`Block`](olm.nn.structure.block.md#olm.nn.structure.block.Block)

Final output projection layer for the Language Model.

Consists of a LayerNorm followed by a Linear projection to the vocabulary size.
Typical structure: LayerNorm -> Linear(vocab_size).

* **Parameters:**
  * **embed_dim** (*int*) – The dimension of the embedding space.
  * **vocab_size** (*int*) – The size of the vocabulary.
  * **bias** (*bool* *,* *optional*) – Whether to include bias in the linear layer. Defaults to False.

#### layers

The normalization and linear layers.

* **Type:**
  nn.ModuleList

### *class* olm.nn.blocks.QKVProjection(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

Bases: `Module`

Computes Query, Key, and Value projections for attention mechanisms.

Applies three separate linear transformations to the input to generate Q, K, and V tensors.
Supports various weight initialization schemes.

#### W_q

Linear layer for Query projection.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### W_k

Linear layer for Key projection.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### W_v

Linear layer for Value projection.

* **Type:**
  [Linear](olm.nn.md#olm.nn.Linear)

#### forward(x: torch.Tensor) → Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

Performs the Q, K, V projections.

* **Parameters:**
  **x** (*torch.Tensor*) – Input tensor of shape (batch, seq_len, dim_in).
* **Returns:**
  A tuple containing (Q, K, V) tensors.
* **Return type:**
  tuple[torch.Tensor, torch.Tensor, torch.Tensor]

### *class* olm.nn.blocks.TransformerBlock(\*args: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any), \*\*kwargs: [Any](olm.data.datasets.base_dataset.md#olm.data.datasets.base_dataset.Any))

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

### Modules

| [`LM`](olm.nn.blocks.LM.md#olm.nn.blocks.LM)(\*args, \*\*kwargs)                                    | A simple Language Model (LM) architecture.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`linear_projections`](olm.nn.blocks.linear_projections.md#module-olm.nn.blocks.linear_projections) |                                              |
| [`output_head`](olm.nn.blocks.output_head.md#module-olm.nn.blocks.output_head)                      |                                              |
| [`transformer_block`](olm.nn.blocks.transformer_block.md#module-olm.nn.blocks.transformer_block)    |                                              |
