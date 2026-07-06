import torch

from olm.nn.attention import (
    GroupedQueryAttention,
    MultiHeadAttentionwithALiBi,
    MultiHeadLatentAttention,
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
