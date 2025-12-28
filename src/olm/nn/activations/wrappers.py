from typing import Optional

import torch
import torch.nn as nn

from olm.core.registry import ACTIVATIONS
from olm.nn.activations.base import ActivationBase


@ACTIVATIONS.register("sigmoid")
class Sigmoid(ActivationBase):
    """Sigmoid activation wrapper."""
    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("tanh")
class Tanh(ActivationBase):
    """Tanh activation wrapper."""
    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("relu")
class ReLU(ActivationBase):
    """ReLU activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("leaky_relu")
class LeakyReLU(ActivationBase):
    """LeakyReLU activation wrapper."""
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("gelu")
class GELU(ActivationBase):
    """GELU activation wrapper."""
    def __init__(self, approximate: str = "none", *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("silu")
@ACTIVATIONS.register("swish")
class SiLU(ActivationBase):
    """SiLU (Swish) activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.SiLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("elu")
class ELU(ActivationBase):
    """ELU activation wrapper."""
    def __init__(self, alpha: float = 1.0, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("selu")
class SELU(ActivationBase):
    """SELU activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.SELU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("prelu")
class PReLU(ActivationBase):
    """PReLU activation wrapper."""
    def __init__(self, num_parameters: int = 1, init: float = 0.25, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        # PReLU parameters need to be on the correct device/dtype
        self.act = nn.PReLU(num_parameters=num_parameters, init=init).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("softplus")
class Softplus(ActivationBase):
    """Softplus activation wrapper."""
    def __init__(self, beta: int = 1, threshold: int = 20, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("mish")
class Mish(ActivationBase):
    """Mish activation wrapper."""
    def __init__(self, inplace: bool = False, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Mish(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("glu")
class GLU(ActivationBase):
    """GLU activation wrapper."""
    def __init__(self, dim: int = -1, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.GLU(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("identity")
class Identity(ActivationBase):
    """Identity activation wrapper."""
    def __init__(self, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


@ACTIVATIONS.register("softmax")
class Softmax(ActivationBase):
    """Softmax activation wrapper."""
    def __init__(self, dim: Optional[int] = None, *, device=None, dtype=None) -> None:
        super().__init__(device=device, dtype=dtype)
        self.act = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
