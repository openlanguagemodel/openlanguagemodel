# `olm.nn.feedforward.swiglu_ffn`

Source: [`src/olm/nn/feedforward/swiglu_ffn.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_ffn.py#L1)

## Classes

### `SwiGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/swiglu_ffn.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_ffn.py#L8)

SwiGLU-based feed-forward network used in modern Transformers (e.g., LLaMA, PaLM).

This layer implements the gated linear unit with Swish (SiLU) activation, which has been
shown to improve performance over standard GELU/ReLU FFNs.

Structure:
    Input
    -> Linear(embed_dim -> 2 * hidden_dim) [Splits into Gate and Value]
    -> SwiGLU(Gate * SiLU(Value))
    -> Linear(hidden_dim -> embed_dim)
    -> Dropout

Args:
    embed_dim (int): The dimension of the input and output.
    hidden_dim (int, optional): The intermediate inner dimension.
        If None, defaults to `int(ff_multiplier * embed_dim)`.
    dropout (float, optional): Dropout probability. Defaults to 0.0.
    bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
    ff_multiplier (float, optional): Multiplier for default hidden dimension. Defaults to 2.5 (commonly 8/3 for SwiGLU).

Attributes:
    up_proj (Linear): Projects and splits input into gate and value parts.
    act (SwiGLU): The activation function.
    down_proj (Linear): Projects back to embedding dimension.
    dropout (nn.Dropout): Dropout layer.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/swiglu_ffn.py:68`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_ffn.py#L68)

Apply SwiGLU feed-forward projection.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.
