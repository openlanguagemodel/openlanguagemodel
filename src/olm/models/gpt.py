"""
GPT Model Definition.

This module will contain the implementation of the GPT (Generative Pre-trained Transformer) architecture.
"""
import torch
from olm.nn.structure import Block
from olm.nn.structure.combinators import Repeat, Residual
from olm.nn.attention import FlashAttention
from olm.nn.feedforward import ClassicFFN
from olm.nn.norms import LayerNorm
from olm.nn.embeddings import Embedding
from olm.nn.embeddings.positional.absolute import AbsolutePositionalEmbedding
from olm.nn.blocks.output_head import OutputHead
from olm.core.registry import MODELS


class GPT2Block():
    """
    A single Transformer block for GPT-2.

    Structure:
        Input -> Residual(LayerNorm -> MHA) -> Residual(LayerNorm -> FFN) -> Output
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        self.block = Block([
            Residual(
                Block([
                    LayerNorm(embed_dim),
                    FlashAttention(embed_dim, num_heads, dropout=dropout, causal=True),
                ])
            ),
            Residual(
                Block([
                    LayerNorm(embed_dim),
                    ClassicFFN(embed_dim, dropout=dropout),
                ])
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block.forward(x)

@MODELS.register("gpt2")
class GPT2(Block):
    """
    GPT-2 124M Model Definition.
    
    Structure:
        Input IDs -> Embedding + PositionalEmbedding -> [GPT2Block] x 12 -> LayerNorm -> OutputHead -> Logits
    """
    def __init__(self):
        # GPT-2 124M Hyperparameters
        vocab_size = 50257
        embed_dim = 768
        num_layers = 12
        num_heads = 12
        max_seq_len = 1024
        dropout = 0.1

        self.stack = Block([
            # 1. Token Embeddings + Positional Embeddings
            Embedding(vocab_size, embed_dim),
            AbsolutePositionalEmbedding(max_seq_len, embed_dim, dropout),
            
            # 2. Transformer Blocks
            Repeat(
                lambda: GPT2Block(embed_dim, num_heads, dropout),
                num_layers
            ),

            # 3. Final LayerNorm (Included in OutputHead)
            
            # 4. Output Head
            OutputHead(embed_dim, vocab_size)
        ])
        
        # Tie weights: OutputHead Linear weight = Embedding weight
        self.blocks[3].blocks[1].weight = self.blocks[0].embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT-2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) containing logits.
        """
        return self.stack.forward(x)