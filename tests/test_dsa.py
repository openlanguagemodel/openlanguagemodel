import torch

from olm.nn.attention import (
    DeepSeekSparseAttention,
    LightningIndexer,
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
        yarn_beta=32.0,
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
