# src/olm/nn/embeddings/positional/sinusoidal.py
import math
import torch
import torch.nn as nn
from typing import Optional
from olm.nn.embeddings.positional.base import PositionalEmbeddingBase


class SinusoidalPositionalEmbedding(PositionalEmbeddingBase):
    """
    Sinusoidal Positional Embedding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Uses fixed sine and cosine functions of different frequencies to encode positions.
    Unlike learned embeddings, these are deterministic and can extrapolate to longer
    sequences than seen during training.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 5000,
        base: int = 10000,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: dimension of the positional embeddings
            max_seq_len: maximum sequence length to precompute embeddings for
            base: base for the geometric progression of wavelengths
            dropout: dropout probability applied to positional embeddings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # precompute positional encodings
        pe = self._build_sinusoidal_encoding(max_seq_len, embed_dim, base)
        self.register_buffer("pe", pe, persistent=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _build_sinusoidal_encoding(
        self, max_seq_len: int, embed_dim: int, base: int
    ) -> torch.Tensor:
        """
        Build sinusoidal positional encoding matrix.

        Returns:
            Tensor of shape (max_seq_len, embed_dim)
        """
        # create position indices
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

        # create dimension indices
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(math.log(base) / embed_dim)
        )

        # initialize encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)

        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cosine to odd indices
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # handle odd embed_dim
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe

    def forward(
        self,
        x: torch.Tensor,
        seq_positions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Apply sinusoidal positional embedding to input tensor x.

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

        if seq_positions is None:
            # use precomputed encodings
            if seq_len > self.max_seq_len:
                # dynamically compute for longer sequences
                pe = self._build_sinusoidal_encoding(seq_len, embed_dim, self.base)
                pe = pe.to(x.device)
            else:
                pe = self.pe[:seq_len]  # (seq_len, embed_dim)

            # add batch dimension and broadcast
            pos_embed = pe.unsqueeze(0)  # (1, seq_len, embed_dim)
        else:
            # custom positions
            max_pos = seq_positions.max().item()
            if max_pos >= self.max_seq_len:
                pe = self._build_sinusoidal_encoding(max_pos + 1, embed_dim, self.base)
                pe = pe.to(x.device)
            else:
                pe = self.pe

            # seq_positions: (b, seq_len)
            # pe: (max_seq_len, embed_dim)
            pos_embed = pe[seq_positions]  # (b, seq_len, embed_dim)

        x = x + pos_embed
        x = self.dropout(x)

        return x
