import pytest
import torch
import torch.nn as nn

from olm.nn.attention import (
    GatedAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    SlidingWindowAttention,
)
from olm.nn.attention.mla import _apply_rope_layout
from olm.nn.blocks import SequentialMultiTokenPredictionHead
from olm.nn.embeddings.positional.rope import (
    RotaryPositionalEmbedding,
    ScaledRotaryPositionalEmbedding,
)
from olm.nn.moe import MoEFeedForward, MoERouter
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import LayerNorm


# --- GQA / sliding-window: separate value head dim and value scaling ---


def test_gqa_supports_separate_value_head_dim():
    attn = GroupedQueryAttention(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=4,
        head_dim=6,
        v_head_dim=4,
        use_rope=False,
    )
    x = torch.randn(2, 4, 16, requires_grad=True)

    out = attn(x)
    out.mean().backward()

    assert attn.v_head_dim == 4
    assert attn.v_proj.out_features == 8
    assert attn.out_proj.in_features == 16
    assert out.shape == x.shape
    assert x.grad is not None


def test_sliding_window_attention_supports_separate_value_head_dim():
    attn = SlidingWindowAttention(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=4,
        window_size=4,
        head_dim=6,
        v_head_dim=4,
        use_rope=False,
    )
    x = torch.randn(2, 4, 16, requires_grad=True)

    out = attn(x)
    out.mean().backward()

    assert attn.v_head_dim == 4
    assert attn.v_proj.out_features == 8
    assert attn.out_proj.in_features == 16
    assert out.shape == x.shape
    assert x.grad is not None


def test_gqa_value_head_dim_matches_manual_attention_on_both_paths():
    # The SDPA path and the eager (softcap / sink) path must agree with a
    # hand-written reference when values are narrower than queries and keys.
    for extra in ({}, {"attn_logit_softcap": 1e6}):
        torch.manual_seed(0)
        attn = GroupedQueryAttention(
            embed_dim=8,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=5,
            head_dim=4,
            v_head_dim=3,
            use_rope=False,
            **extra,
        )
        x = torch.randn(1, 5, 8)

        q = attn.q_proj(x).view(1, 5, 2, 4).transpose(1, 2)
        k = attn.k_proj(x).view(1, 5, 1, 4).transpose(1, 2)
        v = attn.v_proj(x).view(1, 5, 1, 3).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * 4**-0.5
        causal = torch.ones(5, 5, dtype=torch.bool).tril()
        probs = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        expected = attn.out_proj((probs @ v).transpose(1, 2).reshape(1, 5, 6))

        assert torch.allclose(attn(x), expected, atol=1e-5)


def test_gqa_and_sliding_window_apply_value_scale():
    modules = [
        GroupedQueryAttention(
            embed_dim=2,
            num_heads=1,
            num_kv_heads=1,
            max_seq_len=3,
            use_rope=False,
            value_scale=0.5,
        ),
        SlidingWindowAttention(
            embed_dim=2,
            num_heads=1,
            num_kv_heads=1,
            max_seq_len=3,
            window_size=3,
            use_rope=False,
            value_scale=0.5,
        ),
    ]
    x = torch.tensor([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]]])

    for attn in modules:
        with torch.no_grad():
            attn.q_proj.weight.zero_()
            attn.k_proj.weight.zero_()
            attn.v_proj.weight.copy_(torch.eye(2))
            attn.out_proj.weight.copy_(torch.eye(2))

        out = attn(x)
        # Uniform attention over [2, 4, 6] is 4; half of it is 2.
        assert torch.allclose(out[0, -1], torch.tensor([2.0, 0.0]), atol=1e-6)


# --- Gated attention: NoPE, sliding window, biases, mask ---


def _gated(**overrides):
    kwargs = dict(embed_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, max_seq_len=8)
    kwargs.update(overrides)
    return GatedAttention(**kwargs)


def test_gated_attention_supports_local_rope_and_global_nope():
    local = _gated(window_size=3)
    global_nope = _gated(use_rope=False)
    x = torch.randn(2, 8, 16, requires_grad=True)

    output = local(x) + global_nope(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert local.rope is not None
    assert global_nope.rope is None
    assert x.grad is not None


def test_gated_attention_window_limits_visible_keys():
    torch.manual_seed(0)
    attn = _gated(window_size=1)
    full = _gated()
    full.load_state_dict(attn.state_dict())
    x = torch.randn(1, 6, 16)

    # A window of 1 lets each token see only itself, so attention returns v.
    v = attn.v_proj(x).view(1, 6, 2, 4).repeat_interleave(2, dim=2).view(1, 6, 16)
    gate = torch.sigmoid(attn.output_gate(x))
    assert torch.allclose(attn(x), attn.out_proj(v * gate), atol=1e-5)

    # A window at least as long as the sequence is plain causal attention.
    attn.window_size = 6
    assert torch.allclose(attn(x), full(x), atol=1e-6)


def test_gated_attention_applies_user_mask():
    torch.manual_seed(0)
    attn = _gated()
    x = torch.randn(1, 5, 16)

    all_true = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    assert torch.allclose(attn(x, mask=all_true), attn(x), atol=1e-6)

    # Hiding the first two tokens from the last query must match running the
    # last three tokens alone (no RoPE, so positions do not matter).
    nope = _gated(use_rope=False)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 2:, :2] = False
    assert torch.allclose(nope(x, mask=mask)[:, 2:], nope(x[:, 2:]), atol=1e-5)


def test_gated_attention_bias_flags():
    attn = _gated(qkv_bias=True, use_bias=True)
    default = _gated()

    assert all(p.bias is not None for p in (attn.q_proj, attn.k_proj, attn.v_proj))
    assert attn.out_proj.bias is not None
    assert default.q_proj.bias is None and default.out_proj.bias is None


# --- MLA rotary layouts ---


def test_split_half_rope_layout_matches_rotate_half_reference():
    rope = RotaryPositionalEmbedding(8, max_seq_len=6)
    x = torch.randn(2, 6, 3, 8)

    out = _apply_rope_layout(rope, x, interleaved=False)

    # Reference "rotate_half" RoPE: dimension i is paired with i + d/2.
    freqs = torch.outer(torch.arange(6, dtype=torch.float32), rope.inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, :, None, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, :, None, :]
    rotate_half = torch.cat([-x[..., 4:], x[..., :4]], dim=-1)
    assert torch.allclose(out, x * cos + rotate_half * sin, atol=1e-6)
    assert torch.equal(_apply_rope_layout(rope, x, interleaved=True), rope(x))


def test_mla_rope_layout_default_is_interleaved():
    kwargs = dict(
        embed_dim=32,
        num_heads=4,
        max_seq_len=8,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=4,
        v_head_dim=6,
    )
    torch.manual_seed(0)
    default = MultiHeadLatentAttention(**kwargs)
    split_half = MultiHeadLatentAttention(**kwargs, rope_interleaved=False)
    split_half.load_state_dict(default.state_dict())
    x = torch.randn(1, 5, 32)

    assert default.rope_interleaved is True
    assert not torch.allclose(default(x), split_half(x))


# --- YaRN frequencies ---


def test_yarn_keeps_fast_dims_and_interpolates_slow_dims():
    rope = ScaledRotaryPositionalEmbedding(
        8,
        max_seq_len=64,
        scaling_type="yarn",
        scaling_factor=4.0,
        original_max_seq_len=16,
    )
    base = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))

    assert torch.allclose(rope.inv_freq[0], base[0])
    assert torch.allclose(rope.inv_freq[-1], base[-1] / 4.0)


def test_yarn_inv_freq_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for head_dim, original, factor, theta in [
        (8, 16, 4.0, 10000.0),
        (64, 4096, 40.0, 10000.0),
        (128, 8192, 8.0, 500000.0),
    ]:
        config = transformers.LlamaConfig(
            hidden_size=2 * head_dim,
            num_attention_heads=2,
            head_dim=head_dim,
            max_position_embeddings=int(original * factor),
            rope_theta=theta,
            rope_scaling={
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original,
            },
        )
        expected, _ = ROPE_INIT_FUNCTIONS["yarn"](config, None)
        rope = ScaledRotaryPositionalEmbedding(
            head_dim,
            max_seq_len=32,
            base=theta,
            scaling_type="yarn",
            scaling_factor=factor,
            original_max_seq_len=original,
        )

        assert torch.allclose(rope.inv_freq, expected)


# --- MoE routing ---


def test_group_limited_router_selects_experts_from_one_group():
    router = MoERouter(
        embed_dim=1,
        num_experts=8,
        top_k=2,
        num_expert_groups=4,
        top_k_groups=1,
        norm_weights=True,
    )
    with torch.no_grad():
        router.gate.weight.copy_(
            torch.tensor([[10.0], [9.0], [8.0], [7.0], [6.0], [5.0], [4.0], [3.0]])
        )

    indices, weights, _ = router(torch.ones(1, 1, 1))

    assert torch.equal(indices.sort().values, torch.tensor([[[0, 1]]]))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1))


def test_group_limited_router_matches_reference():
    torch.manual_seed(0)
    router = MoERouter(
        embed_dim=16,
        num_experts=16,
        top_k=4,
        scoring_func="sigmoid",
        routing_method="noaux_tc",
        num_expert_groups=4,
        top_k_groups=2,
    )
    with torch.no_grad():
        router.expert_bias.copy_(torch.randn(16) * 0.1)
    x = torch.randn(3, 5, 16)

    indices, weights, logits = router(x)

    scores = torch.sigmoid(logits) + router.expert_bias
    for token, token_scores in enumerate(scores.view(-1, 16)):
        groups = token_scores.view(4, 4)
        group_rank = groups.topk(2, dim=-1).values.sum(dim=-1)
        allowed = group_rank.topk(2).indices
        candidates = torch.cat([torch.arange(4) + 4 * g for g in allowed.tolist()])
        best = candidates[token_scores[candidates].topk(4).indices]
        assert set(indices.view(-1, 4)[token].tolist()) == set(best.tolist())
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 5))


def test_router_rejects_invalid_group_settings():
    with pytest.raises(ValueError, match="num_expert_groups"):
        MoERouter(embed_dim=4, num_experts=6, top_k=2, num_expert_groups=4)
    with pytest.raises(ValueError, match="top_k_groups"):
        MoERouter(embed_dim=4, num_experts=8, num_expert_groups=2, top_k_groups=3)
    with pytest.raises(ValueError, match="top_k"):
        MoERouter(
            embed_dim=4, num_experts=8, top_k=3, num_expert_groups=4, top_k_groups=1
        )


def test_moe_feed_forward_passes_group_settings_to_router():
    moe = MoEFeedForward(
        embed_dim=8,
        expert_cls=SwiGLUFFN,
        num_experts=8,
        top_k=2,
        expert_kwargs={"hidden_dim": 8},
        num_expert_groups=4,
        top_k_groups=2,
    )

    out, _ = moe(torch.randn(2, 3, 8))

    assert out.shape == (2, 3, 8)
    assert moe.router.num_expert_groups == 4
    assert moe.router.top_k_groups == 2


def test_fp32_router_runs_projection_in_float32():
    router = MoERouter(
        embed_dim=4,
        num_experts=4,
        top_k=2,
        fp32_gate=True,
    ).to(dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, weights, logits = router(torch.randn(2, 3, 4, dtype=torch.bfloat16))

    assert logits.dtype == torch.float32
    assert weights.dtype == torch.float32


def test_moe_router_smebu_update_is_centered_and_uses_momentum():
    router = MoERouter(
        embed_dim=8,
        num_experts=4,
        top_k=2,
        scoring_func="sigmoid",
        routing_method="noaux_tc",
    )
    fraction = torch.tensor([0.50, 0.25, 0.15, 0.10])

    router.update_expert_bias_softclamped_momentum_(fraction, momentum=0.5)
    first = router.expert_bias.detach().clone()
    router.update_expert_bias_softclamped_momentum_(fraction, momentum=0.5)
    second = router.expert_bias.detach() - first

    # Reference: u = 5e-4 * tanh(2 * (1/E - f) / (1/E)), centered;
    # m1 = 0.5 * u, m2 = 0.5 * m1 + 0.5 * u.
    update = 5e-4 * torch.tanh(2.0 * (0.25 - fraction) / 0.25)
    update = update - update.mean()
    assert torch.allclose(first, 0.5 * update)
    assert torch.allclose(second, 0.75 * update)
    assert torch.isclose(first.sum(), torch.tensor(0.0), atol=1e-8)
    assert first[0] < 0 < first[-1]


def test_smebu_momentum_is_not_in_state_dict():
    router = MoERouter(embed_dim=8, num_experts=4, routing_method="noaux_tc")

    assert "expert_bias_momentum" not in router.state_dict()
    with pytest.raises(ValueError, match="noaux_tc"):
        MoERouter(embed_dim=8, num_experts=4).update_expert_bias_softclamped_momentum_(
            torch.full((4,), 0.25)
        )


# --- LayerNorm ---


def test_layer_norm_can_keep_scale_without_bias():
    torch.manual_seed(0)
    norm = LayerNorm(8, bias=False)
    with torch.no_grad():
        norm.gamma.normal_()
    reference = nn.LayerNorm(8, bias=False)
    with torch.no_grad():
        reference.weight.copy_(norm.gamma)
    x = torch.randn(2, 4, 8)

    assert norm.beta is None
    assert torch.allclose(norm(x), reference(x), atol=1e-6)


# --- Sequential multi-token prediction ---


def test_sequential_mtp_shares_output_weight_and_preserves_depth_chain():
    output_weight = nn.Parameter(torch.randn(24, 8))
    mtp = SequentialMultiTokenPredictionHead(
        embed_dim=8,
        output_weight=output_weight,
        block_factory=lambda: nn.Sequential(nn.Linear(8, 8), nn.SiLU()),
        num_heads=2,
    )
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    embeddings = torch.randn(2, 6, 8, requires_grad=True)

    logits = mtp(hidden, embeddings)
    sum(item.mean() for item in logits).backward()

    assert [item.shape for item in logits] == [(2, 5, 24), (2, 4, 24)]
    assert mtp.output_weight is output_weight
    assert output_weight.grad is not None
    assert hidden.grad is not None
    assert embeddings.grad is not None


def test_sequential_mtp_depth_uses_shifted_embeddings():
    torch.manual_seed(0)
    mtp = SequentialMultiTokenPredictionHead(
        embed_dim=4,
        output_weight=nn.Parameter(torch.randn(10, 4)),
        block_factory=nn.Identity,
        num_heads=1,
    )
    hidden = torch.randn(1, 5, 4)
    embeddings = torch.randn(1, 5, 4)

    (logits,) = mtp(hidden, embeddings)

    # Position i combines h[i] with the embedding of token i + 1.
    combined = torch.cat(
        [mtp.hidden_norms[0](hidden[:, :4]), mtp.embedding_norms[0](embeddings[:, 1:])],
        dim=-1,
    )
    expected = mtp.projections[0](combined) @ mtp.output_weight.T
    assert torch.allclose(logits, expected, atol=1e-6)
