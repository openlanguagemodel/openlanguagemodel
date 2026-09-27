"""Structural checks for the preset fixes that bring models in line with the
released checkpoints (issues #81 and #83 and the follow-up audit).

Every model here is tiny (vocab 128, width 32) so the file runs in seconds on
CPU. Hyperparameter values of the full-size presets are checked separately,
against the released configs, in ``tests/test_preset_reference_configs.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from olm.models import Gemma2Model, Gemma3Model, MistralSmall3_1_Model, Olmo3Model
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


def _key_blocking_mask(seq_len):
    """Boolean mask that hides key 0 from every query except query 0."""
    mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool)
    mask[..., 1:, 0] = False
    return mask


def _tiny_gemma2():
    return Gemma2Model(128, 32, 64, 2, 4, 2, 8, 16, sliding_window=4)


def _tiny_gemma3():
    return Gemma3Model(128, 32, 64, 6, 4, 2, 8, max_seq_len=16, sliding_window=4)


def _tiny_olmo3():
    return Olmo3Model(128, 32, 64, 4, 4, 2, 8, max_seq_len=16, sliding_window=4)


@pytest.mark.parametrize(
    "build",
    [_tiny_gemma2, _tiny_gemma3, _tiny_olmo3],
    ids=["gemma2", "gemma3", "olmo3"],
)
def test_models_with_custom_blocks_forward_the_attention_mask(build):
    torch.manual_seed(0)
    model = build().eval()
    tokens = torch.randint(0, 128, (1, 8))
    all_true = torch.ones(1, 1, 8, 8, dtype=torch.bool)

    with torch.no_grad():
        plain = model(tokens)
        with_all_true = model(tokens, mask=all_true)
        with_blocking = model(tokens, mask=_key_blocking_mask(8))

    # An all-True mask is a no-op on top of the causal (and window) mask...
    torch.testing.assert_close(with_all_true, plain)
    # ...while hiding key 0 changes every later position, and only those.
    torch.testing.assert_close(with_blocking[:, 0], plain[:, 0])
    assert not torch.allclose(with_blocking[:, 1:], plain[:, 1:])


def test_gemma_feed_forwards_use_tanh_gelu():
    gemma2_block = _tiny_gemma2().blocks[1].stack[0]
    gemma3_block = _tiny_gemma3().transformer_blocks[0]

    assert gemma2_block.mlp.act.approximate == "tanh"
    assert gemma3_block.mlp.act.approximate == "tanh"


def test_gemma3_query_pre_attn_scalar_sets_local_and_global_scale():
    model = Gemma3Model(
        128,
        32,
        64,
        6,
        4,
        2,
        8,
        max_seq_len=16,
        sliding_window=4,
        query_pre_attn_scalar=168.0,
    )
    local_attn = model.transformer_blocks[0].self_attn
    global_attn = model.transformer_blocks[5].self_attn

    assert local_attn.scale == pytest.approx(168.0**-0.5)
    assert global_attn.scale == pytest.approx(168.0**-0.5)
    # Without it, both fall back to head_dim ** -0.5.
    default = _tiny_gemma3()
    assert default.transformer_blocks[0].self_attn.scale == pytest.approx(8**-0.5)
    assert default.transformer_blocks[5].self_attn.scale == pytest.approx(8**-0.5)
