# `olm.nn.feedforward.moe_base`

Source: [`src/olm/nn/feedforward/moe_base.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L1)

## Classes

### `MoEFeedForwardBase(embed_dim: int, expert_cls: Type[torch.nn.modules.module.Module], num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, expert_kwargs: dict = None, **kwargs)`

**Bases:** `olm.nn.feedforward.base.FeedForwardBase`

Source: [`src/olm/nn/feedforward/moe_base.py:47`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L47)

Base class for Mixture of Experts FeedForward networks.

Supports:
- Top-K routing
- Shared experts (always active)
- Dynamic expert instantiation

#### Methods

##### `forward(self, x: torch.Tensor) -> torch.Tensor`

Source: [`src/olm/nn/feedforward/moe_base.py:100`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L100)

Forward pass with MoE routing.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    torch.Tensor: Hidden states shaped ``[batch, seq_len, embed_dim]``.

### `MoERouter(embed_dim: int, num_experts: int, top_k: int = 2)`

**Bases:** `Module`

Source: [`src/olm/nn/feedforward/moe_base.py:10`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L10)

Router for Mixture of Experts.

Routes input tokens to the top-k experts based on learned gate logits.

#### Methods

##### `forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

Source: [`src/olm/nn/feedforward/moe_base.py:22`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/nn/feedforward/moe_base.py#L22)

Route each token to its top-k experts.

Args:
    x (torch.Tensor): Hidden states shaped ``[batch, seq_len, embed_dim]``.

Returns:
    tuple[torch.Tensor, torch.Tensor]: Expert indices and normalized
    routing weights, both shaped ``[batch, seq_len, top_k]``.
