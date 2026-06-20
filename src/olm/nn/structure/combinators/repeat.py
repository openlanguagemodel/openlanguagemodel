from olm.nn.structure.combinators.base import BaseCombinator
import torch.nn as nn
import torch
from typing import Any, Callable

class Repeat(BaseCombinator):
    """
    Repeat a module a fixed number of times in sequence.

    The module function should return a new module instance each call. It is
    used to build ``stack`` during initialization and is not needed for forward
    passes after the modules have been created.

    Args:
        module_func: Callable returning a new module instance.
        num_repeat: Number of times to repeat the module.

    Attributes:
        num_repeat: Number of repeats.
        stack: ModuleList containing the repeated modules.
    """
    def __init__(self, module_func: Callable[[], nn.Module], num_repeat: int):
        """
        Initialize the repeat combinator.

        Args:
            module_func: Callable returning a new module instance.
            num_repeat: Number of times to repeat the module.
        """
        super().__init__()

        self.module = module_func
        self.num_repeat = num_repeat

        self.stack = nn.ModuleList([module_func() for _ in range(num_repeat)])

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["module"] = None
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the repeated modules in sequence.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all repeats.
        """
        for block in self.stack:
            x = block(x)
        return x
