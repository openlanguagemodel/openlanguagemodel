# src/olm/nn/embeddings/positional/alibi.py
import math
import torch
import torch.nn as nn
from typing import Optional
from olm.nn.embeddings.positional.base import PositionalEmbeddingBase


class ALiBiPositionalBias(PositionalEmbeddingBase):
    """
    Attention with Linear Biases (ALiBi) as described in
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    (arXiv 2108.12409).

    Instead of adding positional information to embeddings, ALiBi adds a bias to attention scores
    that is proportional to the distance between query and key positions. This allows the model
    to extrapolate to longer sequences than seen during training.

    The bias is computed as: ``bias[i,j] = -m * |i - j|``
    where m is a head-specific slope.
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        """
        Args:
            num_heads: number of attention heads
            max_seq_len: maximum sequence length to precompute biases for
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # compute head-specific slopes
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

        # crecompute bias matrix for efficiency
        # chape: (num_heads, max_seq_len, max_seq_len)
        alibi_bias = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Compute geometric sequence of slopes for each head.

        For n heads, slopes are: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        # handle non-power-of-2 heads
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # find closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # add extra slopes
            extra_slopes = self._get_slopes(2 * closest_power_of_2).tolist()
            slopes = slopes + extra_slopes[0::2][: num_heads - closest_power_of_2]

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias(self, max_seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
        """
        Build the full ALiBi bias matrix.

        Returns:
            Tensor of shape (num_heads, max_seq_len, max_seq_len)
        """
        # create position matrix: distances[i,j] = |i - j|
        pos = torch.arange(max_seq_len).unsqueeze(0)  # (1, max_seq_len)
        distances = torch.abs(pos.T - pos)  # (max_seq_len, max_seq_len)

        # apply slopes: bias = -m * distance
        # slopes: (num_heads,) -> (num_heads, 1, 1)
        # distances: (max_seq_len, max_seq_len) -> (1, max_seq_len, max_seq_len)
        alibi_bias = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)

        return alibi_bias  # (num_heads, max_seq_len, max_seq_len)

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias for the given query and key sequence lengths.

        Args:
            seq_len_q: length of query sequence
            seq_len_k: length of key sequence (usually same as seq_len_q)
            device: device to place the bias tensor on

        Returns:
            Bias tensor of shape (1, num_heads, seq_len_q, seq_len_k)
            This should be added to attention scores before softmax.
        """
        if seq_len_q > self.max_seq_len or seq_len_k > self.max_seq_len:
            # dynamically compute for longer sequences
            max_len = max(seq_len_q, seq_len_k)
            alibi_bias = self._build_alibi_bias(max_len, self.slopes)
            if device is not None:
                alibi_bias = alibi_bias.to(device)
            bias = alibi_bias[:, :seq_len_q, :seq_len_k]
        else:
            # use precomputed bias
            bias = self.alibi_bias[:, :seq_len_q, :seq_len_k]
            if device is not None and bias.device != device:
                bias = bias.to(device)

        # add batch dimension: (num_heads, seq_len_q, seq_len_k) -> (1, num_heads, seq_len_q, seq_len_k)
        return bias.unsqueeze(0)
