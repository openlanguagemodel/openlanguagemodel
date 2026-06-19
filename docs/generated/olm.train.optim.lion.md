# `olm.train.optim.lion`

Source: [`src/olm/train/optim/lion.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/lion.py#L1)

## Classes

### `Lion(params: Iterable, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False)`

**Bases:** `olm.train.optim.base.OptimizerBase`

Source: [`src/olm/train/optim/lion.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/lion.py#L7)

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

##### `step(self, closure: Callable[[], float] | None = None) -> float | None`

Source: [`.venv/lib/python3.14/site-packages/torch/utils/_contextlib.py:67`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/.venv/lib/python3.14/site-packages/torch/utils/_contextlib.py#L67)

Performs a single optimization step.

Args:
    closure: A closure that reevaluates the model and returns the loss.

Returns:
    Optional loss value if closure is provided.

##### `zero_grad(self, set_to_none: bool = True)`

Source: [`src/olm/train/optim/lion.py:126`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/lion.py#L126)

Sets gradients of all optimized tensors to zero.

Args:
    set_to_none: instead of setting to zero, set the grads to None.
        This is more memory efficient and can slightly improve performance.
