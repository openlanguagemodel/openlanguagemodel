# olm.nn.blocks.LM

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

#### \_\_init_\_(vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int, dropout: float = 0.0, causal: bool = True, ff_multiplier: float = 2.5)

### Methods

| `__init__`(vocab_size, embed_dim, num_heads, ...)   |                                            |
|-----------------------------------------------------|--------------------------------------------|
| `forward`(x)                                        | Apply each block to the input in sequence. |
