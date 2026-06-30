import torch
import torch.nn as nn
from typing import Optional

from olm.core.registry import NORMS
from olm.nn.norms.base import NormBase


@NORMS.register("group_norm")
class GroupNorm(NormBase):
    """
    Group Normalization layer.

    Divides channels into groups and normalizes within each group.
    Used by Ling 2.5 (group_norm_size=8) for stabilizing MoE expert
    routing and attention computation.

    Reference: "Group Normalization" (https://arxiv.org/abs/1803.08494)

    Args:
        d_model (int): Number of channels (features) to normalize.
        num_groups (int): Number of groups to divide the channels into.
            ``d_model`` must be divisible by ``num_groups``.
        eps (float): Small constant for numerical stability.
        affine (bool): If True, learns per-channel scale (gamma) and
            shift (beta) parameters.
        device (torch.device, optional): Target device.
        dtype (torch.dtype, optional): Target data type.
    """

    def __init__(
        self,
        d_model: int,
        num_groups: int = 8,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(d_model, device=device, dtype=dtype)
        if d_model % num_groups != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(
                torch.ones(d_model, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(d_model, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization.

        Args:
            x: Input tensor. Supports shapes:
                - ``[batch, seq_len, d_model]`` (NLP / transformer hidden states)
                - ``[batch, d_model, *]`` (vision / conv features)

        Returns:
            Normalized tensor of the same shape.
        """
        orig_shape = x.shape
        in_dtype = x.dtype

        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)

        x = x.to(torch.float32)
        num_tokens = x.shape[0]
        channels_per_group = self.d_model // self.num_groups

        x = x.view(num_tokens, self.num_groups, channels_per_group)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(num_tokens, self.d_model)

        if self.affine:
            x = x * self.weight + self.bias

        x = x.view(orig_shape)
        return x.to(in_dtype)
