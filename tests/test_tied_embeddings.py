import pytest
import torch
import torch.nn.functional as F

from olm.nn.blocks import LM, OutputHead
from olm.nn.embeddings import Embedding
from olm.nn.structure import load_block


def test_lm_ties_embeddings_by_default():
    model = LM(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=8,
    )

    assert model.blocks[2].weight is model.blocks[0].embedding.weight

    logits = model(torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)

    logits.mean().backward()
    assert model.blocks[0].embedding.weight.grad is not None
    assert model.blocks[2].weight.grad is model.blocks[0].embedding.weight.grad


def test_lm_can_disable_tied_embeddings():
    model = LM(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=8,
        tie_embeddings=False,
    )

    assert model.blocks[2].weight is not model.blocks[0].embedding.weight


def test_output_head_ties_by_default_and_requires_embedding():
    with pytest.raises(ValueError, match="ties weights by default"):
        OutputHead(embed_dim=16, vocab_size=32)


def test_output_head_can_disable_tying():
    embedding = Embedding(vocab_size=32, embedding_dim=16)
    head = OutputHead(
        embed_dim=16,
        vocab_size=32,
        tied_embedding=embedding,
        tie_weights=False,
    )

    assert head.weight is not embedding.embedding.weight


def test_lm_save_load_preserves_tied_embeddings(tmp_path):
    torch.manual_seed(0)
    model = LM(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=8,
    )
    x = torch.randint(0, 32, (2, 8))
    expected = model(x)

    model.save(str(tmp_path / "lm"))
    loaded = load_block(str(tmp_path / "lm"))

    assert loaded.blocks[2].weight is loaded.blocks[0].embedding.weight
    assert torch.allclose(loaded(x), expected)


def test_output_head_uses_tied_embedding_weight():
    embedding = Embedding(vocab_size=32, embedding_dim=16)
    head = OutputHead(embed_dim=16, vocab_size=32, tied_embedding=embedding)
    x = torch.randn(2, 8, 16)

    expected = F.linear(head.blocks[0](x), embedding.embedding.weight)
    assert torch.allclose(head(x), expected)
    assert head.weight is embedding.embedding.weight


def test_output_head_validates_tied_embedding_shape():
    embedding = Embedding(vocab_size=32, embedding_dim=16)

    with pytest.raises(ValueError, match="embedding dimension"):
        OutputHead(embed_dim=8, vocab_size=32, tied_embedding=embedding)

    with pytest.raises(ValueError, match="vocabulary size"):
        OutputHead(embed_dim=16, vocab_size=64, tied_embedding=embedding)
