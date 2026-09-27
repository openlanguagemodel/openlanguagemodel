import torch

from olm.nn.attention import (
    DeepSeekSparseAttention,
    LightningIndexer,
    MultiHeadLatentAttention,
    causal_topk_selection,
    indexer_kl_alignment_loss,
)


def test_lightning_indexer_scores_are_causal():
    torch.manual_seed(0)
    indexer = LightningIndexer(embed_dim=16, num_heads=4, head_dim=8)
    scores = indexer(torch.randn(2, 6, 16))

    assert scores.shape == (2, 6, 6)
    future = torch.ones(6, 6, dtype=torch.bool).triu(1)
    assert torch.isneginf(scores[:, future]).all()


def test_causal_topk_selection_keeps_only_finite_prefix_tokens():
    scores = torch.arange(25, dtype=torch.float32).view(1, 5, 5)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))

    selected_indices, selection_mask = causal_topk_selection(scores, top_k=3)

    assert selected_indices.shape == (1, 5, 3)
    assert selection_mask.shape == (1, 5, 5)
    assert selection_mask[0, 0].sum() == 1
    assert selection_mask[0, 1].sum() == 2
    assert selection_mask[0, 4].sum() == 3
    assert not selection_mask[0].triu(1).any()


def _dsa(**overrides):
    kwargs = dict(
        embed_dim=32,
        num_heads=4,
        max_seq_len=16,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=6,
        q_lora_rank=12,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=3,
    )
    kwargs.update(overrides)
    return DeepSeekSparseAttention(**kwargs)


def test_deepseek_sparse_attention_reuses_mla_and_trains_one_step():
    torch.manual_seed(0)
    attention = _dsa()
    x = torch.randn(2, 8, 32, requires_grad=True)

    out = attention(x)
    stats = attention.last_stats
    assert stats is not None

    target_scores = torch.randn_like(stats.index_scores)
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    target_scores = target_scores.masked_fill(~causal, float("-inf"))
    indexer_loss = indexer_kl_alignment_loss(stats.index_scores, target_scores)
    (out.square().mean() + indexer_loss).backward()

    assert attention.kv_b_proj.out_features == 4 * (6 + 6)
    assert out.shape == x.shape
    assert x.grad is not None
    assert attention.indexer.q_proj.weight.grad is not None
    assert stats.index_scores.shape == (2, 8, 8)
    assert stats.selection_mask[:, -1].sum(dim=-1).max() <= 3
    assert stats.metadata["mode"] == "reference_masked_sdpa"


def test_deepseek_sparse_attention_combines_external_mask():
    attention = _dsa()
    x = torch.randn(1, 6, 32)
    mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    mask[..., 4:, 1] = False

    out = attention(x, mask=mask)

    assert out.shape == x.shape
    assert attention.last_stats is not None
    assert not attention.last_stats.selection_mask[:, 4:, 1].any()


def test_deepseek_sparse_attention_can_use_scaled_rope():
    attention = _dsa(
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,
        original_max_seq_len=4,
        yarn_beta_fast=32.0,
    )

    assert attention.rope.scaling_type == "yarn"
    assert attention.rope.scaling_factor == 4.0
    assert attention.rope.original_max_seq_len == 4


def test_indexer_kl_alignment_loss_supports_sparse_mask():
    torch.manual_seed(0)
    index_scores = torch.randn(2, 5, 5)
    target_scores = torch.randn(2, 5, 5)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    index_scores = index_scores.masked_fill(~causal, float("-inf"))
    target_scores = target_scores.masked_fill(~causal, float("-inf"))
    _, selection_mask = causal_topk_selection(index_scores, top_k=2)

    dense_loss = indexer_kl_alignment_loss(index_scores, target_scores)
    sparse_loss = indexer_kl_alignment_loss(
        index_scores,
        target_scores,
        selection_mask=selection_mask,
    )

    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)


def test_dsa_uses_mla_query_latent_and_produces_alignment_loss():
    attention = _dsa(index_n_heads=2)
    x = torch.randn(2, 8, 32, requires_grad=True)

    output = attention(x, compute_alignment_loss=True)
    stats = attention.last_stats
    assert stats is not None
    assert stats.target_distribution is not None
    assert stats.alignment_loss is not None

    (output.square().mean() + stats.alignment_loss).backward()

    assert attention.indexer.q_proj.in_features == 12
    assert torch.isfinite(stats.alignment_loss)
    assert attention.indexer.q_proj.weight.grad is not None
    assert x.grad is not None


def test_dsa_indexer_input_is_detached_by_default():
    attention = _dsa()
    x = torch.randn(1, 6, 32, requires_grad=True)

    attention(x, compute_alignment_loss=True)
    attention.last_stats.alignment_loss.backward()

    # The KL loss trains the indexer only.
    assert attention.indexer.q_proj.weight.grad is not None
    assert attention.q_a_proj.weight.grad is None
    assert x.grad is None


def test_dsa_indexer_head_weights_stay_fp32_under_autocast():
    attention = _dsa(embed_dim=16, num_heads=2, kv_lora_rank=4, q_lora_rank=8)
    x = torch.randn(1, 5, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        attention(x)

    assert attention.last_stats.index_scores.dtype == torch.float32


def test_dsa_supports_both_rope_dimension_layouts():
    torch.manual_seed(0)
    # A single rotary pair is the same in both layouts, so use two pairs.
    dims = dict(qk_nope_head_dim=4, qk_rope_head_dim=4)
    interleaved = _dsa(**dims, rope_interleaved=True, indexer_rope_interleaved=True)
    split_half = _dsa(**dims, rope_interleaved=False, indexer_rope_interleaved=False)
    split_half.load_state_dict(interleaved.state_dict())
    x = torch.randn(1, 5, 32)

    assert interleaved(x).shape == split_half(x).shape == (1, 5, 32)
    assert not torch.allclose(interleaved(x), split_half(x))


def test_dsa_dense_stage_matches_mla_with_the_same_weights():
    torch.manual_seed(0)
    dsa = _dsa()
    mla = MultiHeadLatentAttention(
        embed_dim=32,
        num_heads=4,
        max_seq_len=16,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=6,
        q_lora_rank=12,
    )
    missing, unexpected = mla.load_state_dict(dsa.state_dict(), strict=False)
    assert not missing
    assert all(key.startswith("indexer.") for key in unexpected)
    x = torch.randn(2, 7, 32)

    assert torch.allclose(dsa(x, use_sparse_attention=False), mla(x), atol=1e-6)
    # Keeping every key makes the sparse stage dense too.
    dsa.index_topk = 7
    assert torch.allclose(dsa(x), mla(x), atol=1e-6)


def test_dsa_supports_dense_indexer_warmup_and_sparse_adaptation():
    attention = _dsa(index_topk=2)
    x = torch.randn(1, 6, 32)

    dense = attention(x, compute_alignment_loss=True, use_sparse_attention=False)
    dense_loss = attention.last_stats.alignment_loss
    sparse = attention(x, compute_alignment_loss=True, use_sparse_attention=True)
    sparse_loss = attention.last_stats.alignment_loss

    assert dense.shape == sparse.shape == x.shape
    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)
    assert not torch.allclose(dense, sparse)
    assert attention.last_stats.metadata["use_sparse_attention"] is True


def test_dsa_accepts_batch_first_3d_mask():
    # batch (3) differs from num_heads (4): a 3-D mask must be read as
    # [batch, seq, seq], not broadcast against the head dimension.
    attention = _dsa()
    x = torch.randn(3, 5, 32)
    mask = torch.ones(3, 5, 5, dtype=torch.bool)
    mask[1, :, 0] = False
    mask[1, 0, 0] = True

    out = attention(x, mask=mask, compute_alignment_loss=True)

    assert out.shape == x.shape
    assert not attention.last_stats.selection_mask[1, 1:, 0].any()
    assert torch.isfinite(attention.last_stats.alignment_loss)
