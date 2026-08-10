import copy

import pytest
import torch

from olm.nn.attention import (
    GroupedQueryAttention,
    MultiHeadAttentionwithALiBi,
    MultiHeadLatentAttention,
    SlidingWindowAttention,
    SparseAttention,
    SparseAttentionwithRoPE,
)
from olm.nn.attention.masks import attention_mask_to_bool


def _mla(**overrides):
    kwargs = dict(
        embed_dim=16,
        num_heads=2,
        max_seq_len=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=6,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return MultiHeadLatentAttention(**kwargs)


def test_mla_forward_backward_shape():
    attn = _mla()
    attn.train()
    x = torch.randn(2, 8, 16, requires_grad=True)

    out = attn(x)
    assert out.shape == (2, 8, 16)

    out.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in attn.parameters() if p.requires_grad)


def test_mla_supports_no_query_lora():
    attn = _mla(q_lora_rank=None)
    assert not hasattr(attn, "q_a_proj")
    assert hasattr(attn, "q_proj")

    out = attn(torch.randn(1, 8, 16))
    assert out.shape == (1, 8, 16)


def test_mla_is_causal_by_default():
    attn = _mla()
    attn.eval()

    x = torch.randn(1, 8, 16)
    out = attn(x)

    perturbed = x.clone()
    perturbed[:, -1, :] += 100.0
    out_perturbed = attn(perturbed)

    # Changing the last token must not affect earlier positions under causal masking.
    assert torch.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-5)


def test_gqa_combines_custom_mask_with_causal_mask():
    attn = GroupedQueryAttention(
        embed_dim=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=4,
        dropout=0.0,
        use_bias=False,
    )

    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.k_proj.weight.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))

    x = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    out = attn(x, mask=mask)

    assert torch.allclose(out[0, 0], x[0, 0], atol=1e-6)


def test_attention_mask_helper_supports_bool_binary_and_additive_masks():
    bool_mask = torch.tensor([[True, False, True]])
    binary_mask = torch.tensor([[1, 0, 1]])
    additive_mask = torch.tensor([[0.0, float("-inf"), 0.0]])

    assert torch.equal(attention_mask_to_bool(bool_mask), bool_mask)
    assert torch.equal(attention_mask_to_bool(binary_mask), bool_mask)
    assert torch.equal(attention_mask_to_bool(additive_mask), bool_mask)


def test_gqa_additive_mask_matches_bool_mask():
    attn = GroupedQueryAttention(
        embed_dim=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=4,
        dropout=0.0,
        use_bias=False,
    )

    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.k_proj.weight.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))

    x = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    bool_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    bool_mask[..., 2, 1] = False
    additive_mask = torch.zeros(1, 1, 3, 3)
    additive_mask = additive_mask.masked_fill(~bool_mask, float("-inf"))

    bool_out = attn(x, mask=bool_mask)
    additive_out = attn(x, mask=additive_mask)

    assert torch.allclose(bool_out, additive_out, atol=1e-6)


def test_gqa_can_disable_rope_for_nope_attention():
    attn = GroupedQueryAttention(
        embed_dim=8,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        use_rope=False,
    )

    assert attn.rope is None
    out = attn(torch.randn(2, 4, 8))
    assert out.shape == (2, 4, 8)


def test_gqa_supports_partial_rope():
    attn = GroupedQueryAttention(
        embed_dim=8,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        partial_rotary_factor=0.5,
    )

    assert attn.rope.rotary_dim == 2
    out = attn(torch.randn(2, 4, 8))
    assert out.shape == (2, 4, 8)


def test_gqa_attention_sink_diverts_probability_mass_to_zero_value_sink():
    attn = GroupedQueryAttention(
        embed_dim=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=1,
        use_rope=False,
        use_attention_sink=True,
        use_bias=False,
    )

    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.k_proj.weight.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))
        attn.attention_sink.zero_()

    x = torch.tensor([[[2.0, 4.0]]])
    out = attn(x)

    assert torch.allclose(out, x * 0.5, atol=1e-6)


def test_sliding_window_attention_respects_external_mask():
    attn = SlidingWindowAttention(
        embed_dim=2,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=4,
        window_size=4,
        use_rope=False,
        use_bias=False,
    )

    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.k_proj.weight.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))

    x = torch.tensor([[[2.0, 0.0], [20.0, 0.0], [6.0, 0.0]]])
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    mask[..., 2, 1] = False

    out = attn(x, mask=mask)

    assert torch.allclose(out[0, 2], torch.tensor([4.0, 0.0]), atol=1e-6)


def test_sliding_window_attention_supports_partial_rope_and_sink():
    attn = SlidingWindowAttention(
        embed_dim=8,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=4,
        partial_rotary_factor=0.5,
        use_attention_sink=True,
    )

    assert attn.rope.rotary_dim == 2
    assert attn.attention_sink.shape == (2,)
    out = attn(torch.randn(2, 4, 8))
    assert out.shape == (2, 4, 8)


def test_alibi_combines_custom_mask_with_causal_mask():
    attn = MultiHeadAttentionwithALiBi(
        embed_dim=2,
        num_heads=1,
        dropout=0.0,
        bias=False,
        causal=True,
        max_seq_len=4,
    )

    with torch.no_grad():
        attn.q_proj.weight.zero_()
        attn.k_proj.weight.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))

    x = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    out = attn(x, mask=mask)

    assert torch.allclose(out[0, 0], x[0, 0], atol=1e-6)


def _sparse(**overrides):
    kwargs = dict(embed_dim=16, num_heads=2, window=4, dropout=0.0)
    kwargs.update(overrides)
    return SparseAttention(**kwargs)


def _sparse_rope(**overrides):
    kwargs = dict(embed_dim=16, num_heads=2, max_seq_len=8, window=4, dropout=0.0)
    kwargs.update(overrides)
    return SparseAttentionwithRoPE(**kwargs)


@pytest.mark.parametrize("backend", ["auto", "sdpa"])
def test_sparse_attention_cpu_forward_backward(backend):
    attn = _sparse(backend=backend)
    attn.train()
    x = torch.randn(2, 8, 16, requires_grad=True)

    out = attn(x)
    assert out.shape == (2, 8, 16)

    out.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in attn.parameters() if p.requires_grad)


@pytest.mark.parametrize("backend", ["auto", "sdpa"])
def test_sparse_attention_with_rope_cpu_forward_backward(backend):
    attn = _sparse_rope(backend=backend)
    attn.train()
    x = torch.randn(2, 8, 16, requires_grad=True)

    out = attn(x)
    assert out.shape == (2, 8, 16)

    out.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in attn.parameters() if p.requires_grad)


@pytest.mark.parametrize("attn_factory", [_sparse, _sparse_rope])
def test_sparse_attention_auto_backend_resolves_to_sdpa_on_cpu(attn_factory):
    attn = attn_factory(backend="auto")
    assert attn._resolve_backend(torch.device("cpu")) == "sdpa"


@pytest.mark.parametrize("attn_factory", [_sparse, _sparse_rope])
def test_sparse_attention_explicit_flex_backend_raises_clear_error_on_cpu(
    attn_factory,
):
    attn = attn_factory(backend="flex")
    x = torch.randn(1, 8, 16)

    with pytest.raises(RuntimeError, match="only supported on CUDA"):
        attn(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("attn_factory", [_sparse, _sparse_rope])
def test_sparse_attention_flex_matches_sdpa_on_cuda(attn_factory):
    attn_sdpa = attn_factory(backend="sdpa").cuda()
    attn_flex = copy.deepcopy(attn_sdpa)
    attn_flex.backend = "flex"

    x_sdpa = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
    x_flex = x_sdpa.detach().clone().requires_grad_(True)

    out_sdpa = attn_sdpa(x_sdpa)
    out_flex = attn_flex(x_flex)
    assert torch.allclose(out_sdpa, out_flex, atol=1e-3, rtol=1e-3)

    out_sdpa.mean().backward()
    out_flex.mean().backward()
    assert torch.allclose(x_sdpa.grad, x_flex.grad, atol=1e-3, rtol=1e-3)
