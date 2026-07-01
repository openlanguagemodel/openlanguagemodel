import inspect
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


def _forward_module(module: nn.Module, x: torch.Tensor, **kwargs):
    """
    Call ``module(x, **filtered_kwargs)``, forwarding only the keyword
    arguments whose names appear in the module's ``forward`` signature.

    This lets combinators propagate ``mask`` (and future kwargs like
    ``position_ids``) through the block chain without crashing on
    sub-modules that don't accept them (e.g. LayerNorm, FFN).
    """
    if not kwargs:
        return module(x)

    sig = inspect.signature(module.forward)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return module(x, **kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in params}
    if accepted:
        return module(x, **accepted)
    return module(x)


class BaseCombinator(nn.Module, ABC):
    """
    Abstract base class for combinator modules.

    Subclasses implement ``forward`` to define how inputs are combined.
    All combinators accept ``**kwargs`` and forward them (e.g. ``mask``)
    to sub-modules that declare matching parameters.
    """
    def __init__(self):
        """Initialize the combinator base."""
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the combinator output from an input tensor.

        Args:
            x: Input tensor.
            **kwargs: Extra arguments (e.g. ``mask``) forwarded to
                sub-modules that accept them.

        Returns:
            Output tensor produced by the combinator.
        """
        pass
