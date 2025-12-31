# src/olm/train/optim/adamw.py
import torch
from torch.optim import AdamW as TorchAdamW
from typing import Optional, Callable, Tuple


class AdamW(TorchAdamW):
    """
    AdamW optimizer with decoupled weight decay regularization.

    This is a wrapper around PyTorch's built-in AdamW implementation from
    "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
    Unlike the original Adam, weight decay is applied directly to the parameters
    rather than being added to the gradient.

    This implementation is commonly used for training large language models and
    transformers, offering better generalization than standard Adam.

    Note: This class inherits from PyTorch's AdamW which ultimately inherits from
    torch.optim.Optimizer, maintaining compatibility with our OptimizerBase interface.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square
            (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
        amsgrad: whether to use the AMSGrad variant (default: False)
        maximize: maximize the params based on the objective, instead of minimizing (default: False)
        fused: whether to use the fused implementation (default: None, auto-detect)

    Example:
        >>> model = nn.Linear(10, 5)
        >>> optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> loss = model(input).sum()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        fused: Optional[bool] = None,
    ):
        # Use PyTorch's built-in AdamW implementation
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=None,  # Let PyTorch decide
            capturable=False,
            differentiable=False,
            fused=fused,
        )
