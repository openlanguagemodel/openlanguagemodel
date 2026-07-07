import torch

from olm.models.builders import (
    build_global_nope_block,
    build_local_swa_block,
    build_moe_transformer_block,
    dense_first_moe_schedule,
    local_global_attention_schedule,
)
from olm.nn.attention import GroupedQueryAttention


def test_dense_first_moe_schedule():
    assert dense_first_moe_schedule(5, 2) == ["dense", "dense", "moe", "moe", "moe"]


def test_local_global_attention_schedule():
    assert local_global_attention_schedule(8, local_layers=3, global_layers=1) == [
        "local",
        "local",
        "local",
        "global",
        "local",
        "local",
        "local",
        "global",
    ]


def test_global_nope_block_forward_shape():
    block = build_global_nope_block(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        hidden_dim=32,
    )
    assert block.attention.rope is None

    x = torch.randn(2, 8, 16)
    out = block(x)

    assert out.shape == x.shape


def test_local_swa_block_forward_shape():
    block = build_local_swa_block(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        window_size=4,
        hidden_dim=32,
        partial_rotary_factor=0.5,
    )
    x = torch.randn(2, 8, 16)
    out = block(x)

    assert out.shape == x.shape


def test_moe_transformer_block_records_router_stats():
    attention = GroupedQueryAttention(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
    )
    block = build_moe_transformer_block(
        embed_dim=16,
        attention=attention,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        expert_hidden_dim=32,
        scoring_func="sigmoid",
        routing_method="noaux_tc",
        use_router_bias=True,
    )
    x = torch.randn(2, 8, 16)
    out = block(x)

    assert out.shape == x.shape
    assert block.last_router_logits is not None
    assert block.last_router_stats is not None


def test_parallel_local_block_backward():
    block = build_local_swa_block(
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        window_size=4,
        hidden_dim=32,
        parallel=True,
    )
    x = torch.randn(2, 8, 16, requires_grad=True)
    loss = block(x).mean()
    loss.backward()

    assert x.grad is not None
