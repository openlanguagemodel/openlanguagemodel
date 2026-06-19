# `olm.nn.feedforward.moe_base`

## Classes

### `MoEFeedForwardBase(embed_dim: int, expert_cls: Type[torch.nn.modules.module.Module], num_experts: int = 8, num_shared_experts: int = 0, top_k: int = 2, expert_kwargs: dict = None, **kwargs)`

Base class for Mixture of Experts FeedForward networks.

Supports:
- Top-K routing
- Shared experts (always active)
- Dynamic expert instantiation

#### Methods

- `forward(self, x: torch.Tensor) -> torch.Tensor`
  Forward pass with MoE routing.

### `MoERouter(embed_dim: int, num_experts: int, top_k: int = 2)`

Router for Mixture of Experts.

Routes input tokens to the top-k experts based on learned gate logits.

#### Methods

- `forward(self, x: torch.Tensor)`
  Define the computation performed at every call.
