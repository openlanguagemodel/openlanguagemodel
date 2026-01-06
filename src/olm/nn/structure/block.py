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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply each block to the input in sequence.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all blocks have been applied.
        """
        for block in self.blocks:
            x = block(x)
        return x

    def save(self, path: str, tokenizer: TokenizerBase=None) -> None:
        if os.path.exists(path):
            raise Exception("Path already exists")
        os.makedirs(path)
        torch.save(self, os.path.join(path, "model.pt"))
        if tokenizer != None:
            tokenizer.save(os.path.join(path, "tokenizer"))

def load(path: str) -> Union["Block", tuple]:
    obj = torch.load(os.path.join(path, "model.pt"), weights_only=False)
    if os.path.exists(os.path.join(path, "tokenizer")):
        tokenizertype = open(os.path.join(path, "tokenizer", "type"), "r").read().strip()
        tokenizer = None
        if tokenizertype == "HFTokenizer":
            tokenizer = HFTokenizer.load(os.path.join(path, "tokenizer"))

        return obj, tokenizer
    return obj

load_model = load
load_block = load
