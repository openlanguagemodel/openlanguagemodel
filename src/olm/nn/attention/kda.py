import torch
import torch.nn.functional as F

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.attention.gated_deltanet import GatedDeltaNet


class KimiDeltaAttention(GatedDeltaNet):
    """
    Kimi Delta Attention (KDA), the linear-attention layer of Kimi Linear.

    KDA refines Gated DeltaNet's delta-rule state update by replacing the
    single scalar forget gate per head with a fine-grained, per-channel
    gate: ``S_t = Diag(alpha_t) @ S_{t-1} + beta_t * k_t @ v_t^T`` where
    ``alpha_t in R^{d_k}`` instead of a scalar. This lets each key channel
    forget the recurrent state at its own rate. Everything else (the causal
    conv, QK-norm + L2-normalization, output gate) is unchanged from
    ``GatedDeltaNet``.

    Interleaved with Multi-head Latent Attention in a 3:1 ratio in Kimi Linear.

    Reference: "Kimi Linear: An Expressive, Efficient Attention Architecture"
    (arXiv:2510.26692)

    Args:
        embed_dim: Model hidden dimension.
        num_heads: Number of KDA heads (shared for keys and values).
        head_dim: Dimension per head.
        conv_kernel_size: Kernel for the causal Conv1d (default 4).
        dropout: Dropout on the output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 32,
        head_dim: int = 128,
        conv_kernel_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(
            embed_dim,
            num_key_heads=num_heads,
            num_value_heads=num_heads,
            key_head_dim=head_dim,
            value_head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        # Fine-grained gate: one forget value per key channel, not per head.
        self.alpha_proj = Linear(embed_dim, num_heads * head_dim, bias=True)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Parallel-mode forward (for prefill / training).

        Identical to ``GatedDeltaNet.forward`` except ``alpha`` is a
        per-channel vector rather than a per-head scalar.

        Args:
            x: ``[batch, seq_len, embed_dim]``

        Returns:
            ``[batch, seq_len, embed_dim]``
        """
        B, N, _ = x.shape

        x_conv = self.conv(x)
        x_conv = F.silu(x_conv)

        alpha = torch.sigmoid(
            self.alpha_proj(x_conv).view(B, N, self.num_key_heads, self.key_head_dim)
        )
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
            alpha_h = alpha[:, :, h]  # [B, N, key_head_dim] -- per-channel gate
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
                a_t = alpha_h[:, t].unsqueeze(-1)  # [B, key_head_dim, 1]
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
