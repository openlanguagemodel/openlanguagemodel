import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from olm.nn.torch_nn_wrappers import Linear
from olm.nn.norms import RMSNorm


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction (MTP) module.

    Predicts ``num_heads`` future tokens in a single forward pass to
    accelerate speculative decoding at inference time.  During training
    each prediction head contributes an auxiliary cross-entropy loss.

    Architecture per head:
        - Embed previous-head hidden state + next-position embedding
        - RMSNorm -> small transformer layer (single attention + FFN)
          or a simple linear projection (``use_transformer=False``)
        - Shared or per-head LM-head projection to vocabulary

    Used by Step 3.5 Flash (MTP-3), MiniMax M2.5 (3 modules), and
    Qwen3.5 (1 module).

    Args:
        embed_dim: Model hidden dimension.
        vocab_size: Vocabulary size for the prediction heads.
        num_heads: Number of future tokens to predict (e.g. 3 for MTP-3).
        share_lm_head: If True, all heads share a single LM projection.
        rms_norm_eps: Epsilon for normalization layers.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_heads: int = 3,
        share_lm_head: bool = True,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.share_lm_head = share_lm_head

        self.norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=rms_norm_eps) for _ in range(num_heads)]
        )

        self.projections = nn.ModuleList(
            [Linear(embed_dim * 2, embed_dim, bias=False) for _ in range(num_heads)]
        )

        if share_lm_head:
            self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        else:
            self.lm_heads = nn.ModuleList(
                [Linear(embed_dim, vocab_size, bias=False) for _ in range(num_heads)]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Produce logits for the next ``num_heads`` future positions.

        Args:
            hidden_states: Final hidden states from the main model,
                ``[batch, seq_len, embed_dim]``.
            token_embeddings: Token embedding matrix output for the input
                sequence, ``[batch, seq_len, embed_dim]``.

        Returns:
            List of ``num_heads`` logit tensors, each
            ``[batch, seq_len - i - 1, vocab_size]``.
        """
        B, N, D = hidden_states.shape
        predictions = []
        prev_hidden = hidden_states

        for i in range(self.num_heads):
            offset = i + 1
            if offset >= N:
                break

            shifted_emb = token_embeddings[:, offset:]
            truncated_hidden = prev_hidden[:, : N - offset]

            combined = torch.cat([truncated_hidden, shifted_emb], dim=-1)
            h = self.projections[i](combined)
            h = self.norms[i](h)

            if self.share_lm_head:
                logits = self.lm_head(h)
            else:
                logits = self.lm_heads[i](h)

            predictions.append(logits)
            prev_hidden = h

        return predictions
