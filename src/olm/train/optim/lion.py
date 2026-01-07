# src/olm/train/optim/lion.py
import torch
from typing import List, Optional, Callable, Tuple, Iterable, Dict, Any
from .base import OptimizerBase


class Lion(OptimizerBase):
    """
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
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            use_triton=use_triton,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("use_triton", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Optional loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Decoupled weight decay (AdamW-style)
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Lion update rule:
                # 1. Interpolate: update = beta1 * m_t + (1 - beta1) * g_t
                # 2. Apply sign: theta_t+1 = theta_t - lr * sign(update)
                # 3. Update momentum: m_t+1 = beta2 * m_t + (1 - beta2) * g_t

                # Compute interpolated gradient
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)

                # Apply sign-based parameter update
                p.add_(torch.sign(update), alpha=-group["lr"])

                # Update exponential moving average (momentum)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """
        Sets gradients of all optimized tensors to zero.

        Args:
            set_to_none: instead of setting to zero, set the grads to None.
                This is more memory efficient and can slightly improve performance.
        """
        super().zero_grad(set_to_none=set_to_none)
