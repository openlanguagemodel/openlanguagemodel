import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from olm.nn.activations.elu import ELU
from olm.nn.activations.gelu import GELU
from olm.nn.activations.geglu import GeGLU
from olm.nn.activations.glu import GLU
from olm.nn.activations.liglu import LiGLU
from olm.nn.activations.reglu import ReGLU
from olm.nn.activations.relu import ReLU
from olm.nn.activations.selu import SELU
from olm.nn.activations.softmax import Softmax
from olm.nn.activations.swiglu import SwiGLU
from olm.nn.activations.swish import Swish
from olm.nn.activations.silu import SiLU
from olm.nn.activations.tanh import Tanh


@pytest.mark.parametrize(
    "activation_cls",
    [ReLU, ELU, GELU, Swish, SiLU, SELU, Tanh, Softmax],
)
def test_pointwise_activation_preserves_shape(activation_cls):
    activation = activation_cls()
    x = torch.randn(2, 3, 4)
    y = activation(x)
    assert y.shape == x.shape


def test_softmax_normalization():
    activation = Softmax(dim=-1)
    x = torch.randn(2, 3, 5)
    y = activation(x)
    sums = y.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


@pytest.mark.parametrize(
    "activation_cls",
    [GLU, GeGLU, LiGLU, ReGLU, SwiGLU],
)
def test_glu_variants_halve_last_dim(activation_cls):
    activation = activation_cls()
    x = torch.randn(2, 6, 8)
    y = activation(x)
    assert y.shape[-1] == x.shape[-1] // 2
    assert y.shape[:-1] == x.shape[:-1]