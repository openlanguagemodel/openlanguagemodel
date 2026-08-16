import torch
import torch.nn as nn
from typing import Optional


class HeadGate(nn.Module):
    """
    Per-head sigmoid gate applied to attention output.

    For each attention head *i* and position, computes:

        ``o_i = sigmoid(w_i^T x) * attn_i(x)``

    where ``w_i`` is a learnable vector of size ``embed_dim``.  This
    suppresses uninformative heads at minimal parameter cost (~1 vector
    per head).  Used by Step 3.5 Flash on both full and sliding-window
    attention layers.

    Reference: Step 3.5 Flash technical report, Section 2.1.

    Args:
        embed_dim: Model hidden dimension (input to the gate).
        num_heads: Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.gate_proj = nn.Parameter(torch.zeros(num_heads, embed_dim))
        nn.init.normal_(self.gate_proj, std=0.02)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pre-attention hidden states ``[batch, seq_len, embed_dim]``.
            attn_out: Post-attention output ``[batch, seq_len, num_heads, head_dim]``
                or ``[batch, num_heads, seq_len, head_dim]``.

        Returns:
            Gated attention output with the same shape as ``attn_out``.
        """
        gate = torch.sigmoid(torch.einsum("bse,he->bsh", x, self.gate_proj))

        if attn_out.dim() == 4:
            if attn_out.shape[1] == self.gate_proj.shape[0]:
                gate = gate.transpose(1, 2).unsqueeze(-1)
            else:
                gate = gate.unsqueeze(-1)

        return attn_out * gate
