# `olm.nn.blocks.transformer_block`

## Classes

### `TransformerBlock(embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0, causal: bool = False, ff_multiplier: float = 2.5)`

A single Transformer block containing Multi-Head Attention and a FeedForward Network.

This block implements the standard Transformer architecture with pre-normalization,
Rotary Positional Embeddings (RoPE), and SwiGLU activation in the feedforward layer.
It supports causal masking for autoregressive modeling.

Structure:
    Input -> LayerNorm -> MHA(RoPE) -> Residual -> LayerNorm -> SwiGLU FFN -> Residual -> Output

Args:
    embed_dim (int): The dimension of the embedding space (d_model).
    num_heads (int): Number of attention heads. verify that embed_dim % num_heads == 0.
    max_seq_len (int): Maximum sequence length supported by the model (for RoPE).
    dropout (float, optional): Dropout probability for attention and FFN. Defaults to 0.0.
    causal (bool, optional): Whether to apply causal masking in attention. Defaults to False.
    ff_multiplier (float, optional): Multiplier for the hidden dimension of the FFN.
        Commonly 4.0 (standard) or 8/3 (SwiGLU). Defaults to 2.5.

Attributes:
    layers (nn.ModuleList): The sequential list of layers within the block.
