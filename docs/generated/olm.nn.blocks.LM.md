# `olm.nn.blocks.LM`

Source: [`src/olm/nn/blocks/LM.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/LM.py#L1)

## Classes

### `LM(vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int, dropout: float = 0.0, causal: bool = True, ff_multiplier: float = 2.5, tie_embeddings: bool = True)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/nn/blocks/LM.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/LM.py#L9)

GPT-style causal language model assembled from OLM blocks.

``LM`` is the small, configurable model used throughout the beginner
examples. It consists of a token embedding, ``num_layers`` repeated
``TransformerBlock`` modules, and an ``OutputHead`` that projects hidden
states back to vocabulary logits. The output projection reuses the input
embedding matrix by default.

**Structure**

``input_ids`` -> ``Embedding`` -> ``TransformerBlock`` x N ->
``OutputHead`` -> logits.

**Forward**

Accepts integer token IDs with shape ``[batch, seq_len]`` and returns
logits with shape ``[batch, seq_len, vocab_size]``. The inherited
``Block.forward`` applies each submodule sequentially.

**Parameters**

- `vocab_size` (`int`): Size of the vocabulary.
- `embed_dim` (`int`): Dimension of the embeddings and hidden states.
- `num_heads` (`int`): Number of attention heads in Transformer blocks.
- `num_layers` (`int`): Number of Transformer blocks.
- `max_seq_len` (`int`): Maximum sequence length for the model.
- `dropout` (`float, optional`): Dropout probability. Defaults to 0.0.
- `causal` (`bool, optional`): Whether to use causal masking. Defaults to True.
- `ff_multiplier` (`float, optional`): Multiplier for FFN hidden dimension. Defaults to 2.5.
- `tie_embeddings` (`bool, optional`): Whether the output head should reuse the input embedding matrix. Defaults to True.

**Attributes**

- `blocks` (`nn.ModuleList`): ``[embedding, transformer_stack, output_head]``.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

**Parameters**

- `x`: Input tensor.

**Returns**

Output tensor after all blocks have been applied.
