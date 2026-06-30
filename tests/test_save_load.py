import torch
import torch.nn as nn
import pytest

from olm.nn.structure import Block, load_block


def test_block_save_load_roundtrip(tmp_path):
    save_path = tmp_path / "block"
    block = Block([nn.Linear(10, 10), nn.ReLU()])
    x = torch.randn(2, 10)

    expected = block(x)
    block.save(str(save_path))

    loaded = load_block(str(save_path))
    actual = loaded(x)

    assert torch.allclose(expected, actual)


def test_block_save_can_update_existing_directory(tmp_path):
    save_path = tmp_path / "block"
    block = Block([nn.Linear(4, 4)])

    block.save(str(save_path))
    block.save(str(save_path))

    loaded = load_block(str(save_path))
    assert isinstance(loaded, Block)


def test_block_load_requires_trusted_artifact(tmp_path):
    save_path = tmp_path / "block"
    block = Block([nn.Linear(4, 4)])
    block.save(str(save_path))

    with pytest.raises(ValueError, match="trust the artifact"):
        load_block(str(save_path), trusted=False)
