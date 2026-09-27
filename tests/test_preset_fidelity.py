"""Structural checks for the preset fixes that bring models in line with the
released checkpoints (issues #81 and #83 and the follow-up audit).

Every model here is tiny (vocab 128, width 32) so the file runs in seconds on
CPU. Hyperparameter values of the full-size presets are checked separately,
against the released configs, in ``tests/test_preset_reference_configs.py``.
"""

import pytest
import torch
import torch.nn.functional as F

import olm.models.minimax as minimax_package
from olm.models import (
    GPT2Model,
    Gemma2Model,
    Gemma3Model,
    Ling2_5_Model,
    MiniMaxM2_5_Model,
    MistralSmall3_1_Model,
    OPTModel,
    Olmo3Model,
    Step3_5_Flash_Model,
)
from olm.nn.activations import GeGLU
from olm.nn.attention import LightningAttention, MultiHeadLatentAttention
from olm.nn.embeddings.positional.rope import (
    PartialRotaryPositionalEmbedding,
    RotaryPositionalEmbedding,
)
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


def test_minimax_package_exports_both_families():
    assert minimax_package.__all__ == [
        "MiniMaxM2Model",
        "MiniMaxM2",
        "MiniMaxM2_5_Model",
        "MiniMaxM2_5_230B",
    ]
    namespace = {}
    exec("from olm.models.minimax import *", namespace)
    assert {"MiniMaxM2Model", "MiniMaxM2"} <= namespace.keys()


def test_minimax_m2_5_applies_partial_rope():
    model = MiniMaxM2_5_Model(
        vocab_size=128,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_mtp_modules=1,
    )
    rope = model.transformer_blocks[0].self_attn.rope

    # rotary_dim=64 of head_dim=128 in the release -> half of each head.
    assert isinstance(rope, PartialRotaryPositionalEmbedding)
    assert rope.rotary_dim == 4
    assert model(torch.randint(0, 128, (1, 8))).shape == (1, 8, 128)


def _tiny_step3_5():
    # One period of the released schedule plus a second full layer:
    # full, sliding, sliding, sliding, full; dense then MoE.
    return Step3_5_Flash_Model(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_layers=5,
        full_attn_heads=4,
        sliding_attn_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
        window_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_mtp_heads=1,
        moe_layers={3, 4},
    )


def test_step3_5_applies_partial_rope_on_full_attention_layers_only():
    blocks = _tiny_step3_5().transformer_blocks
    full_rope = blocks[0].self_attn.rope
    sliding_rope = blocks[1].self_attn.rope

    # partial_rotary_factors is [0.5, 1.0, 1.0, 1.0] * 12 in the release.
    assert isinstance(full_rope, PartialRotaryPositionalEmbedding)
    assert full_rope.rotary_dim == 4
    assert isinstance(sliding_rope, RotaryPositionalEmbedding)
    assert isinstance(blocks[4].self_attn.rope, PartialRotaryPositionalEmbedding)
    assert blocks[0].self_attn.rope.base == 5000000.0
    assert blocks[1].self_attn.rope.base == 10000.0


class _ZeroFFN(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["full", "sliding"])
@pytest.mark.parametrize("seq_len", [4, 8], ids=["seq_eq_heads", "seq_ne_heads"])
def test_step3_5_head_gate_scales_heads_before_the_output_projection(
    layer_idx, seq_len
):
    torch.manual_seed(0)
    block = _tiny_step3_5().transformer_blocks[layer_idx]
    block.ffn = _ZeroFFN()
    x = torch.randn(2, seq_len, 32)

    with torch.no_grad():
        h = block.attn_norm(x)
        heads = block.self_attn(h).view(2, seq_len, block.num_heads, -1)
        gate = torch.sigmoid(h @ block.head_gate.gate_proj.T)
        gated = (heads * gate.unsqueeze(-1)).reshape(2, seq_len, -1)
        expected = x + block.o_proj(gated)

        torch.testing.assert_close(block(x), expected)


def test_step3_5_tiny_model_runs_with_mtp():
    model = _tiny_step3_5()
    tokens = torch.randint(0, 128, (1, 8))
    logits, mtp_logits = model.forward_with_mtp(tokens)

    assert logits.shape == (1, 8, 128)
    assert len(mtp_logits) == 1
    # Layers 0-2 are dense; the MoE layers record their router logits.
    assert model.transformer_blocks[2].last_router_logits is None
    assert model.transformer_blocks[3].last_router_logits is not None


def test_ling2_5_places_mla_last_in_each_group_and_on_the_tail():
    model = Ling2_5_Model(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_layers=6,
        num_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_seq_len=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_group_size=4,
        first_k_dense=1,
    )
    attn_types = [type(block.self_attn) for block in model.transformer_blocks]

    # BailingMoeV2_5DecoderLayer: MLA iff (i + 1) % group == 0, or i is past
    # the last full group. The released 80-layer model has MLA at 7, 15, ... 79.
    assert attn_types == [
        LightningAttention,
        LightningAttention,
        LightningAttention,
        MultiHeadLatentAttention,
        MultiHeadLatentAttention,
        MultiHeadLatentAttention,
    ]
    assert model(torch.randint(0, 128, (1, 8))).shape == (1, 8, 128)


def test_gpt2_uses_tanh_gelu():
    # config.json: activation_function="gelu_new", the tanh approximation.
    model = GPT2Model(
        vocab_size=128, embed_dim=32, num_layers=1, num_heads=4, max_seq_len=16
    )
    ffn = model.blocks[1].stack[0].blocks[1].block.blocks[1]

    assert ffn.act.act.approximate == "tanh"


def test_opt_layer_norms_use_torch_default_eps():
    # HF OPT builds every LayerNorm with nn.LayerNorm's default eps (1e-5).
    model = OPTModel(128, 32, 64, 1, 4, dropout=0.0)
    block = model.blocks[3].stack[0]

    assert block.blocks[0].block.blocks[0].eps == 1e-5
    assert block.blocks[1].block.blocks[0].eps == 1e-5
    assert model.blocks[4].eps == 1e-5
