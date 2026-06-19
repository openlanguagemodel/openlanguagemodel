# `olm.nn.blocks.transformer_block`

Source: [`src/olm/nn/blocks/transformer_block.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/transformer_block.py#L1)

## Classes

### `TransformerBlock(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, ff_multiplier: float = 2.5)`

**Bases:** `olm.nn.structure.block.Block`

Source: [`src/olm/nn/blocks/transformer_block.py:9`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/blocks/transformer_block.py#L9)

Pre-norm transformer block for causal language modeling.

The block contains two residual branches: LayerNorm + FlashAttention with
RoPE, followed by LayerNorm + SwiGLU feed-forward network. It is the default
repeated block used by ``LM``.

Structure:
    ``x`` -> residual(LayerNorm, attention) -> residual(LayerNorm, FFN).

Forward:
    Accepts hidden states with shape ``[batch, seq_len, embed_dim]`` and
    returns hidden states with the same shape.

Args:
    embed_dim (int): The dimension of the embedding space (d_model).
    num_heads (int): Number of attention heads. verify that embed_dim % num_heads == 0.
    max_seq_len (int): Maximum sequence length supported by the model (for RoPE).
    dropout (float, optional): Dropout probability for attention and FFN. Defaults to 0.0.
    causal (bool, optional): Whether to apply causal masking in attention. Defaults to False.
    ff_multiplier (float, optional): Multiplier for the hidden dimension of the FFN.
        Commonly 4.0 (standard) or 8/3 (SwiGLU). Defaults to 2.5.

Attributes:
    blocks (nn.ModuleList): Sequential OLM block structure.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `Block`)

Source: [`src/olm/nn/structure/block.py:26`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/structure/block.py#L26)

Apply each block to the input in sequence.

Args:
    x: Input tensor.

Returns:
    Output tensor after all blocks have been applied.
