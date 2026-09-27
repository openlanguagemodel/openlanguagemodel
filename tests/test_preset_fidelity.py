"""Structural checks for the preset fixes that bring models in line with the
released checkpoints (issues #81 and #83 and the follow-up audit).

Every model here is tiny (vocab 128, width 32) so the file runs in seconds on
CPU. Hyperparameter values of the full-size presets are checked separately,
against the released configs, in ``tests/test_preset_reference_configs.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from olm.models import MistralSmall3_1_Model
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


def test_mistral_small3_1_head_dim_sets_projection_widths():
    # Mistral Small 3.1 uses 128-wide heads with embed_dim=5120 and 32 heads
    # (#81); this is the same geometry at toy scale.
    model = MistralSmall3_1_Model(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        head_dim=4,
    )
    attn = model.blocks[1].stack[0].blocks[0].block.blocks[1]

    assert attn.head_dim == 4
    assert attn.q_proj.weight.shape == (16, 32)
    assert attn.k_proj.weight.shape == (8, 32)
    assert attn.out_proj.weight.shape == (32, 16)
    assert model(torch.randint(0, 128, (1, 8))).shape == (1, 8, 128)


def test_mistral_small3_1_head_dim_defaults_to_embed_dim_over_heads():
    model = MistralSmall3_1_Model(128, 32, 64, 1, 4, 2, 16)
    attn = model.blocks[1].stack[0].blocks[0].block.blocks[1]

    assert attn.head_dim == 8
