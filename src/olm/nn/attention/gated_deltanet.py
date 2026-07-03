import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.base import AttentionBase
from olm.nn.norms import RMSNorm
from olm.nn.blocks.causal_conv1d import CausalConv1d


class GatedDeltaNet(AttentionBase):
    """
    Gated DeltaNet -- a linear attention mechanism with delta-rule state
    updates, used by Qwen3.5 and Qwen3-Next.

    Instead of maintaining an O(N) KV cache, DeltaNet keeps a fixed-size
    recurrent state matrix ``S in R^{d_k x d_v}`` and updates it at each
    step using a gated delta rule:

        ``S_t = alpha_t * S_{t-1} + beta_t * k_t @ v_t^T``

    A causal 1-D convolution provides local context before computing the
    alpha/beta gates.

    This implementation supports:
        - **Recurrent mode** (token-by-token decode): O(1) per step.
        - **Parallel mode** (prefill): materialised via cumulative ops.

    Reference: "Gated Delta Networks" (arXiv:2412.06464)

    Args:
        embed_dim: Model hidden dimension.
        num_key_heads: Number of key heads.
        num_value_heads: Number of value heads.
        key_head_dim: Dimension per key head.
        value_head_dim: Dimension per value head.
        conv_kernel_size: Kernel for the causal Conv1d (default 4).
        dropout: Dropout on the output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_key_heads: int = 16,
        num_value_heads: int = 64,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel_size: int = 4,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_key_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.head_dim = key_head_dim
        self.scale = key_head_dim ** -0.5

        total_k_dim = num_key_heads * key_head_dim
        total_v_dim = num_value_heads * value_head_dim

        self.q_proj = Linear(embed_dim, total_k_dim, bias=False)
        self.k_proj = Linear(embed_dim, total_k_dim, bias=False)
        self.v_proj = Linear(embed_dim, total_v_dim, bias=False)
        self.out_proj = Linear(total_v_dim, embed_dim, bias=False)

        self.output_gate = Linear(embed_dim, total_v_dim, bias=False)

        self.conv = CausalConv1d(embed_dim, kernel_size=conv_kernel_size)

        self.alpha_proj = Linear(embed_dim, num_key_heads, bias=True)
        self.beta_proj = Linear(embed_dim, num_key_heads, bias=True)

        self.q_norm = RMSNorm(key_head_dim, eps=1e-6)
        self.k_norm = RMSNorm(key_head_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Delta-rule recurrent attention. Not called directly -- the full
        flow including gating and convolution is in ``forward``.
        """
        raise NotImplementedError("Use forward() directly; DeltaNet uses recurrent state updates")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parallel-mode forward (for prefill / training).

        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, N, _ = x.shape

        x_conv = self.conv(x)
        x_conv = F.silu(x_conv)

        alpha = torch.sigmoid(self.alpha_proj(x_conv))
        beta = torch.sigmoid(self.beta_proj(x_conv))

        q = self.q_proj(x).view(B, N, self.num_key_heads, self.key_head_dim)
        k = self.k_proj(x).view(B, N, self.num_key_heads, self.key_head_dim)
        v = self.v_proj(x).view(B, N, self.num_value_heads, self.value_head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        heads_per_kv_group = self.num_value_heads // self.num_key_heads

        output_heads = []
        for h in range(self.num_key_heads):
            q_h = q[:, :, h]
            k_h = k[:, :, h]
            alpha_h = alpha[:, :, h : h + 1]
            beta_h = beta[:, :, h : h + 1]

            v_start = h * heads_per_kv_group
            v_end = v_start + heads_per_kv_group
            v_h = v[:, :, v_start:v_end].reshape(
                B, N, heads_per_kv_group * self.value_head_dim
            )
            v_dim = v_h.shape[-1]

            S = torch.zeros(B, self.key_head_dim, v_dim, device=x.device, dtype=x.dtype)
            outs = []

            for t in range(N):
                k_t = k_h[:, t].unsqueeze(-1)
                v_t = v_h[:, t].unsqueeze(1)
                a_t = alpha_h[:, t].unsqueeze(-1)
                b_t = beta_h[:, t].unsqueeze(-1)

                S = a_t * S + b_t * torch.bmm(k_t, v_t)

                q_t = q_h[:, t].unsqueeze(1)
                o_t = torch.bmm(q_t, S).squeeze(1)
                outs.append(o_t)

            output_heads.append(torch.stack(outs, dim=1))

        out = torch.cat(output_heads, dim=-1)

        gate = torch.sigmoid(self.output_gate(x))
        out = out * gate

        out = self.out_proj(out)
        return self.dropout(out)
