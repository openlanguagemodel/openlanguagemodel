from .base import BaseCombinator
from .parallel import Parallel
from .repeat import Repeat
from .residual import Residual

try:
    from .merge_functions import Add, MergeFunction, Concat, Matmul
except Exception:
    Add = MergeFunction = Concat = Matmul = None

__all__ = [
    "BaseCombinator",
    "Parallel",
    "Repeat",
    "Residual",
    "Add",
    "MergeFunction",
    "Concat",
    "Matmul",
]
