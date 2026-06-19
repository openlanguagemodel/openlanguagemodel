# `olm.train.optim.adamw`

Source: [`src/olm/train/optim/adamw.py:1`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/adamw.py#L1)

## Classes

### `AdamW(params, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False, fused: bool | None = None)`

**Bases:** `AdamW`

Source: [`src/olm/train/optim/adamw.py:7`](https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/train/optim/adamw.py#L7)

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
