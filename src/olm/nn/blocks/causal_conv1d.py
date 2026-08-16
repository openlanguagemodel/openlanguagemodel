import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Depthwise causal 1-D convolution.

    Applies a causal (left-padded) convolution independently per channel,
    giving each position a local context window of ``kernel_size`` tokens
    without leaking future information.

    Used by Qwen3.5 / Qwen3-Next Gated DeltaNet layers (kernel_size=4)
    and Mamba-style state-space models.

    Args:
        channels: Number of input channels (applied depthwise).
        kernel_size: Size of the convolutional kernel.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[batch, seq_len, channels]``

        Returns:
            Causally convolved tensor with the same shape.
        """
        x = x.transpose(1, 2)  # [B, C, N]
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)  # [B, N, C]
