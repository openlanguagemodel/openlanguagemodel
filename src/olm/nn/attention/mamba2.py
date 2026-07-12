import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.blocks.causal_conv1d import CausalConv1d
from olm.nn.norms import RMSNorm


class Mamba2Mixer(nn.Module):
    """
    Mamba-2 selective state-space mixer (SSD).

    Projects the input into a gate ``z``, a conv branch ``x, B, C`` (passed
    through a causal depthwise convolution), and a per-head timestep ``dt``.
    Each head keeps a fixed-size recurrent state ``S in R^{head_dim x state_size}``
    updated with an input-dependent, per-head scalar decay:

        ``dA_t = exp(dt_t * A)``
        ``S_t = dA_t * S_{t-1} + dt_t * outer(x_t, B_t)``
        ``y_t = S_t @ C_t + D * x_t``

    The output is gated (``y * silu(z)``) then RMSNorm'd -- equivalent to the
    "gated RMSNorm" used in the reference implementation -- before the final
    output projection. Used by the Nemotron-H / Nemotron 3 family as the
    linear-time alternative to attention.

    Reference: "Transformers are SSMs: Generalized Models and Efficient
    Algorithms Through Structured State Space Duality" (Mamba-2), arXiv:2405.21060

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of SSM heads.
        head_dim: Per-head dimension (``num_heads * head_dim`` is the inner width).
        state_size: Per-head recurrent state size (``N``).
        n_groups: Number of groups sharing a ``B``/``C`` projection (GQA-style).
        conv_kernel_size: Kernel for the causal Conv1d applied to ``x, B, C``.
        time_step_floor: Minimum value ``dt`` is clamped to for stability.
        rms_norm_eps: Epsilon for the pre-output-projection RMSNorm.
        bias: Whether to use bias in the in/out projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        state_size: int = 128,
        n_groups: int = 1,
        conv_kernel_size: int = 4,
        time_step_floor: float = 1e-4,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.n_groups = n_groups
        self.time_step_floor = time_step_floor
        self.d_inner = num_heads * head_dim

        self.conv_dim = self.d_inner + 2 * n_groups * state_size
        self.in_proj = Linear(
            embed_dim,
            2 * self.d_inner + 2 * n_groups * state_size + num_heads,
            bias=bias,
        )
        self.conv = CausalConv1d(self.conv_dim, kernel_size=conv_kernel_size)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))

        self.norm = RMSNorm(self.d_inner, eps=rms_norm_eps)
        self.out_proj = Linear(self.d_inner, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Sequential-scan forward (for prefill / training).

        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B_batch, N, _ = x.shape

        proj = self.in_proj(x)
        z, xBC, dt = torch.split(
            proj, [self.d_inner, self.conv_dim, self.num_heads], dim=-1
        )

        xBC = F.silu(self.conv(xBC))
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_groups * self.state_size,
                self.n_groups * self.state_size,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias).clamp(min=self.time_step_floor)
        A = -torch.exp(self.A_log)  # [num_heads], negative decay rate

        x_ssm = x_ssm.view(B_batch, N, self.num_heads, self.head_dim)
        B_ssm = B_ssm.view(B_batch, N, self.n_groups, self.state_size)
        C_ssm = C_ssm.view(B_batch, N, self.n_groups, self.state_size)

        heads_per_group = self.num_heads // self.n_groups
        # Repeat B/C across the heads sharing each group (GQA-style).
        B_ssm = B_ssm.repeat_interleave(heads_per_group, dim=2)
        C_ssm = C_ssm.repeat_interleave(heads_per_group, dim=2)

        state = torch.zeros(
            B_batch,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=x.device,
            dtype=x.dtype,
        )
        outs = []
        for t in range(N):
            dt_t = dt[:, t]  # [B, num_heads]
            dA_t = torch.exp(dt_t * A)  # [B, num_heads]
            x_t = x_ssm[:, t]  # [B, num_heads, head_dim]
            B_t = B_ssm[:, t]  # [B, num_heads, state_size]
            C_t = C_ssm[:, t]  # [B, num_heads, state_size]

            update = torch.einsum("bh,bhd,bhn->bhdn", dt_t, x_t, B_t)
            state = dA_t[:, :, None, None] * state + update

            y_t = torch.einsum("bhdn,bhn->bhd", state, C_t)
            y_t = y_t + self.D[None, :, None] * x_t
            outs.append(y_t)

        y = torch.stack(outs, dim=1).reshape(B_batch, N, self.d_inner)

        y = self.norm(y * F.silu(z))
        return self.out_proj(y)
