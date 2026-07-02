import torch.nn as nn
import torch
import os
from typing import List, Union
from olm.data.tokenization import TokenizerBase, HFTokenizer


class Block(nn.Module):
    """
    Lightweight sequential container for composable submodules.

    Similar to ``nn.Sequential``, but exposes the underlying list for
    inspection or dynamic manipulation by higher-level builders.

    Args:
        blocks: Ordered list of modules applied to the input in sequence.

    Attributes:
        blocks: ModuleList storing the ordered blocks.
    """

    def __init__(self, blocks: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply each block to the input in sequence.

        Extra keyword arguments (e.g. ``mask``) are forwarded to every
        sub-module whose ``forward`` signature accepts them.

        Args:
            x: Input tensor.
            **kwargs: Forwarded to sub-modules that accept them.

        Returns:
            Output tensor after all blocks have been applied.
        """
        from olm.nn.structure.combinators.base import _forward_module

        for block in self.blocks:
            x = _forward_module(block, x, **kwargs)
        return x

    def save(self, path: str, tokenizer: TokenizerBase = None) -> None:
        """
        Save this block as a trusted local artifact.

        OLM preserves the complete Python module object so custom architectures can
        round-trip without a separate config format. Only load artifacts you created
        yourself or otherwise trust.
        """
        if os.path.exists(path) and not os.path.isdir(path):
            raise NotADirectoryError(f"Save path exists and is not a directory: {path}")
        os.makedirs(path, exist_ok=True)
        torch.save(self, os.path.join(path, "model.pt"))
        if tokenizer != None:
            tokenizer.save(os.path.join(path, "tokenizer"))


def load(path: str, *, trusted: bool = True) -> Union["Block", tuple]:
    """
    Load a block saved by ``Block.save``.

    ``Block.save`` stores the full Python module object so arbitrary custom
    architectures can be restored. This uses Python pickle through PyTorch and is
    only safe for trusted local artifacts.

    Args:
        path: Directory produced by ``Block.save``.
        trusted: Must be ``True``. Pass this explicitly in security-sensitive code
            to make the trusted-artifact boundary visible.
    """
    if not trusted:
        raise ValueError(
            "OLM model directories contain pickled Python modules and can only be "
            "loaded when you trust the artifact."
        )

    obj = torch.load(os.path.join(path, "model.pt"), weights_only=False)
    if os.path.exists(os.path.join(path, "tokenizer")):
        tokenizertype = (
            open(os.path.join(path, "tokenizer", "type"), "r").read().strip()
        )
        tokenizer = None
        if tokenizertype == "HFTokenizer":
            tokenizer = HFTokenizer.load(os.path.join(path, "tokenizer"))

        return obj, tokenizer
    return obj


load_model = load
load_block = load
