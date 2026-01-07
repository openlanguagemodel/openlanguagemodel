# src/olm/nn/embeddings/positional/absolute.py
import torch
import torch.nn as nn
from typing import Optional
from olm.nn.embeddings.positional.base import PositionalEmbeddingBase


class AbsolutePositionalEmbedding(PositionalEmbeddingBase):
    """
    Absolute (Learned) Positional Embedding.

    This is the standard positional embedding used in the original Transformer paper
    and models like GPT-2. It learns a separate embedding vector for each position
    in the sequence, up to a maximum sequence length.

    These embeddings are typically added to token embeddings before passing through
    the transformer blocks.
    """

    def __init__(self, max_seq_len: int, embed_dim: int, dropout: float = 0.0):
        """
        Args:
            max_seq_len: maximum sequence length to support
            embed_dim: dimension of the positional embeddings
            dropout: dropout probability applied to positional embeddings
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, seq_positions: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Apply absolute positional embedding to input tensor x.

        Args:
            x: shape (batch_size, seq_len, embed_dim) - token embeddings
            seq_positions: optional tensor of shape (batch_size, seq_len) with position indices.
                If None, assumes positions are 0..seq_len-1 for each batch.

        Returns:
            Tensor of same shape as x, with positional embeddings added.
        """
        b, seq_len, embed_dim = x.shape
        assert (
            embed_dim == self.embed_dim
        ), f"Expected embed_dim={self.embed_dim}, got {embed_dim}"

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} greater than max_seq_len {self.max_seq_len}"
            )

        if seq_positions is None:
            # default: positions 0, 1, 2, ..., seq_len-1
            positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
            positions = positions.unsqueeze(0).expand(b, seq_len)  # (b, seq_len)
        else:
            positions = seq_positions

        # get positional embeddings and add to input
        pos_embed = self.pos_embedding(positions)  # (b, seq_len, embed_dim)
        x = x + pos_embed
        x = self.dropout(x)

        return x
