from .base import FeedForwardBase
from .classic_ffn import ClassicFFN
from .swiglu_ffn import SwiGLUFFN

__all__ = [
    "ClassicFFN",
    "SwiGLUFFN",
]