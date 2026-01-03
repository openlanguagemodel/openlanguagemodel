"""
GPT Model Definition.

This module will contain the implementation of the GPT (Generative Pre-trained Transformer) architecture.
"""

import torch
import torch.nn as nn
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import FlashAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.absolute import AbsolutePositionalEmbedding
from olm.nn.blocks.output_head import OutputHead


class GPT2Block(nn.Module):
    """
    A single Transformer block for GPT-2.

    Structure:
        Input -> Residual(LayerNorm -> MHA) -> Residual(LayerNorm -> FFN) -> Output
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.block = Block(
            [
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim),
                            FlashAttention(
                                embed_dim, num_heads, dropout=dropout, causal=True
                            ),
                        ]
                    )
                ),
                Residual(
                    Block(
                        [
                            LayerNorm(embed_dim),
                            ClassicFFN(embed_dim, dropout=dropout),
                        ]
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GPT2(nn.Module):
    """
    Base GPT-2 Model Architecture.

    Structure:
        Input IDs -> Embedding + PositionalEmbedding -> [GPT2Block] x L -> LayerNorm -> OutputHead -> Logits
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = Block(
            [
                # 1. Token Embeddings + Positional Embeddings
                Embedding(vocab_size, embed_dim),
                AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
                # 2. Transformer Blocks
                Repeat(lambda: GPT2Block(embed_dim, num_heads, dropout), num_layers),
                # 3. Final LayerNorm (Included in OutputHead)
                # 4. Output Head
                OutputHead(embed_dim, vocab_size),
            ]
        )

        # Tie weights: OutputHead Linear weight = Embedding weight
        self.model.blocks[3].blocks[1].weight = self.model.blocks[0].embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GPT2Small(GPT2):
    """
    GPT-2 Small (124M) version.
    """

    def __init__(self):
        super().__init__(embed_dim=768, num_layers=12, num_heads=12)


class GPT2Medium(GPT2):
    """
    GPT-2 Medium (355M) version.
    """

    def __init__(self):
        super().__init__(embed_dim=1024, num_layers=24, num_heads=16)


class GPT2Large(GPT2):
    """
    GPT-2 Large (774M) version.
    """

    def __init__(self):
        super().__init__(embed_dim=1280, num_layers=36, num_heads=20)


class GPT2XL(GPT2):
    """
    GPT-2 XL (1.5B) version.
    """

    def __init__(self):
        super().__init__(embed_dim=1600, num_layers=48, num_heads=25)
