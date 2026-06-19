# `olm.train.optim.lion`

## Classes

### `Lion(params: Iterable, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False)`

Lion optimizer (EvoLved Sign Momentum).

Implements the Lion algorithm from "Symbolic Discovery of Optimization Algorithms"
(Chen et al., 2023). Lion uses only the sign of the gradient for updates,
making it more memory-efficient than Adam while often achieving better performance.

Key differences from Adam:
- Uses sign of interpolated gradient for updates (memory efficient)
- Single momentum buffer instead of two (m and v in Adam)
- Typically requires smaller learning rates (1/3 to 1/10 of AdamW)
- Larger weight decay (3-10x that of AdamW)

Args:
    params: iterable of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-4, typically 3-10x smaller than AdamW)
    betas: coefficients used for computing running averages (default: (0.9, 0.99))
    weight_decay: weight decay coefficient (default: 0.0)
    use_triton: whether to use Triton kernel for faster computation (default: False)

Example:
    >>> model = nn.Linear(10, 5)
    >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
    >>> optimizer.zero_grad()
    >>> loss = model(input).sum()
    >>> loss.backward()
    >>> optimizer.step()

#### Methods

- `step(self, closure: Callable[[], float] | None = None) -> float | None`
  Performs a single optimization step.
- `zero_grad(self, set_to_none: bool = True)`
  Sets gradients of all optimized tensors to zero.
