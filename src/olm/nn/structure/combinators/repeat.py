from olm.nn.structure.combinators.base import BaseCombinator
import torch.nn as nn
import torch
from typing import Callable

# note that module_func has to be a lambda function
class Repeat(BaseCombinator):
    def __init__(self, module_func: Callable[[], nn.Module], num_repeat: int):
        super().__init__()

        self.module = module_func
        self.num_repeat = num_repeat

        self.stack = nn.ModuleList([module_func() for _ in range(num_repeat)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.stack:
            x = block(x)
        return x