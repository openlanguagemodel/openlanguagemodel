"""Structural checks for the preset fixes that bring models in line with the
released checkpoints (issues #81 and #83 and the follow-up audit).

Every model here is tiny (vocab 128, width 32) so the file runs in seconds on
CPU. Hyperparameter values of the full-size presets are checked separately,
against the released configs, in ``tests/test_preset_reference_configs.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from olm.nn.activations import GeGLU
from olm.nn.feedforward import GeGLUFFN


def test_geglu_defaults_to_exact_gelu():
    x = torch.randn(2, 3, 8)
    value, gate = x.chunk(2, dim=-1)

    torch.testing.assert_close(GeGLU()(x), value * F.gelu(gate))


def test_geglu_tanh_approximation_matches_gelu_pytorch_tanh():
    x = torch.randn(2, 3, 8) * 3
    value, gate = x.chunk(2, dim=-1)
    expected = value * F.gelu(gate, approximate="tanh")

    torch.testing.assert_close(GeGLU(approximate="tanh")(x), expected)
    assert not torch.allclose(GeGLU()(x), expected)


def test_geglu_rejects_unknown_approximation():
    with pytest.raises(ValueError, match="approximate"):
        GeGLU(approximate="sigmoid")


def test_geglu_ffn_forwards_approximation_to_its_activation():
    assert GeGLUFFN(16, hidden_dim=32).act.approximate == "none"
    assert GeGLUFFN(16, hidden_dim=32, approximate="tanh").act.approximate == "tanh"
