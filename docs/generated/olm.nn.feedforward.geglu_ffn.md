# `olm.nn.feedforward.geglu_ffn`

Source: [`src/olm/nn/feedforward/geglu_ffn.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_ffn.py#L1)

## Classes

### `GeGLUFFN(embed_dim: int, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 4.0)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/geglu_ffn.py:8`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_ffn.py#L8)

Feed-Forward Network using GeGLU activation.

Implements: x = DownProj(GeGLU(UpProj(x))).
UpProj expands to 2 * hidden_dim to support splitting for the gate.

**Parameters**

- `embed_dim` (`int`): Input dimension.
- `hidden_dim` (`int, optional`): Hidden dimension. Defaults to 4 * embed_dim if None.
- `dropout` (`float, optional`): Dropout probability. Defaults to 0.0.
- `bias` (`bool, optional`): Whether to usage bias in linear layers. Defaults to True.
- `ff_multiplier` (`float, optional`): Expansion factor if hidden_dim is None. Defaults to 4.0.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/geglu_ffn.py:54`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/geglu_ffn.py#L54)

Apply GeGLU feed-forward projection.

**Parameters**

- `x` (`torch.Tensor`): Hidden states shaped ``[batch, seq_len, embed_dim]``.

**Returns**

- `torch.Tensor`: Hidden states shaped ``[batch, seq_len, embed_dim]``.
