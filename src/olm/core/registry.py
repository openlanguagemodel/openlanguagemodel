# src/olm/core/registry.py
from typing import Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self):
        self._f: Dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., T]):
            self._f[name] = fn
            return fn

        return deco

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._f:
            raise KeyError(f"Unknown component: {name}. Registered: {list(self._f)}")
        return self._f[name]


ACTIVATIONS = Registry()
NORMS = Registry()
ATTN = Registry()
POS_EMB = Registry()
MOE = Registry()
MODELS = Registry()
LOSSES = Registry()
