# `olm.nn.feedforward.swiglu_moe`

Source: [`src/olm/nn/feedforward/swiglu_moe.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_moe.py#L1)

## Classes

### `SwiGLUMoEFFN(embed_dim: int, num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, hidden_dim: int = None, dropout: float = 0.0, bias: bool = True, ff_multiplier: float = 2.5, **kwargs)`

**Bases:** `olm.nn.feedforward.moe_base.MoEFeedForwardBase`

Source: [`src/olm/nn/feedforward/swiglu_moe.py:4`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/swiglu_moe.py#L4)

Mixture of Experts version of SwiGLUFFN.

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor` (inherited from `MoEFeedForwardBase`)

Source: [`src/olm/nn/feedforward/moe_base.py:100`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L100)

Forward pass with MoE routing.

**Parameters**

- `x` (`torch.Tensor`): Hidden states shaped ``[batch, seq_len, embed_dim]``.

**Returns**

- `torch.Tensor`: Hidden states shaped ``[batch, seq_len, embed_dim]``.
