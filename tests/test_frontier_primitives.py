import torch
import torch.nn as nn

from olm.nn.attention import DeepSeekSparseAttention, GatedAttention
from olm.nn.blocks import SequentialMultiTokenPredictionHead
from olm.nn.moe import MoERouter
from olm.nn.norms import LayerNorm


def test_layer_norm_can_keep_scale_without_bias():
    norm = LayerNorm(8, bias=False)
    output = norm(torch.randn(2, 4, 8))

    assert output.shape == (2, 4, 8)
    assert norm.gamma is not None
    assert norm.beta is None


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


def test_gated_attention_supports_local_rope_and_global_nope():
    local = GatedAttention(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        window_size=3,
    )
    global_nope = GatedAttention(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        use_rope=False,
    )
    x = torch.randn(2, 8, 16, requires_grad=True)

    output = local(x) + global_nope(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert local.rope is not None
    assert global_nope.rope is None
    assert x.grad is not None


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


def test_dsa_uses_mla_query_latent_and_produces_alignment_loss():
    attention = DeepSeekSparseAttention(
        embed_dim=32,
        num_heads=4,
        max_seq_len=16,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=6,
        q_lora_rank=12,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=3,
    )
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


def test_dsa_indexer_head_weights_stay_fp32_under_autocast():
    attention = DeepSeekSparseAttention(
        embed_dim=16,
        num_heads=2,
        max_seq_len=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
    )
    x = torch.randn(1, 5, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        attention(x)

    assert attention.last_stats.index_scores.dtype == torch.float32


def test_dsa_supports_both_rope_dimension_layouts():
    common = dict(
        embed_dim=16,
        num_heads=2,
        max_seq_len=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
    )
    x = torch.randn(1, 5, 16)

    interleaved = DeepSeekSparseAttention(
        **common,
        rope_interleaved=True,
        indexer_rope_interleaved=True,
    )
    split_half = DeepSeekSparseAttention(
        **common,
        rope_interleaved=False,
        indexer_rope_interleaved=False,
    )

    assert interleaved(x).shape == split_half(x).shape == (1, 5, 16)
    assert not torch.allclose(interleaved(x), split_half(x))


def test_dsa_supports_dense_indexer_warmup_and_sparse_adaptation():
    attention = DeepSeekSparseAttention(
        embed_dim=16,
        num_heads=2,
        max_seq_len=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
    )
    x = torch.randn(1, 6, 16)

    dense = attention(
        x,
        compute_alignment_loss=True,
        use_sparse_attention=False,
    )
    dense_loss = attention.last_stats.alignment_loss
    sparse = attention(
        x,
        compute_alignment_loss=True,
        use_sparse_attention=True,
    )
    sparse_loss = attention.last_stats.alignment_loss

    assert dense.shape == sparse.shape == x.shape
    assert torch.isfinite(dense_loss)
    assert torch.isfinite(sparse_loss)
    assert not torch.allclose(dense, sparse)
    assert attention.last_stats.metadata["use_sparse_attention"] is True
