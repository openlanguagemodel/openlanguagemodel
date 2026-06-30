import torch

from olm.nn.embeddings.positional.rope import (
    PartialRotaryPositionalEmbedding,
    PartialScaledRotaryPositionalEmbedding,
    RotaryPositionalEmbedding,
    ScaledRotaryPositionalEmbedding,
)


def test_rope_cache_is_lazy_and_grows_to_used_context():
    rope = RotaryPositionalEmbedding(head_dim=128, max_seq_len=128_000)
    assert rope.emb_sin.numel() == 0
    assert rope.emb_cos.numel() == 0

    x = torch.randn(2, 16, 4, 128)
    y = rope(x)

    assert y.shape == x.shape
    assert rope.emb_sin.shape == (16, 1, 64)
    assert rope.emb_cos.shape == (16, 1, 64)


def test_partial_rope_cache_uses_max_position_when_positions_are_given():
    rope = PartialRotaryPositionalEmbedding(
        head_dim=64,
        rotary_percentage=0.5,
        max_seq_len=128,
    )
    x = torch.randn(1, 4, 2, 64)
    positions = torch.tensor([[0, 7, 12, 31]])

    y = rope(x, positions)

    assert y.shape == x.shape
    assert rope.emb_sin.shape[0] == 32


def test_scaled_rope_variants_are_lazy():
    rope = ScaledRotaryPositionalEmbedding(head_dim=32, max_seq_len=1024)
    partial = PartialScaledRotaryPositionalEmbedding(
        head_dim=32,
        rotary_percentage=0.5,
        max_seq_len=1024,
    )
    x = torch.randn(1, 8, 2, 32)

    assert rope.emb_sin.numel() == 0
    assert partial.emb_sin.numel() == 0
    assert rope(x).shape == x.shape
    assert partial(x).shape == x.shape
    assert rope.emb_sin.shape[0] == 8
    assert partial.emb_sin.shape[0] == 8


def test_scaled_rope_xpos_expands_pairwise_scale_to_head_dim():
    rope = ScaledRotaryPositionalEmbedding(
        head_dim=8,
        max_seq_len=16,
        scaling_type="xpos",
    )
    x = torch.randn(2, 4, 3, 8)

    y = rope(x)

    assert y.shape == x.shape
