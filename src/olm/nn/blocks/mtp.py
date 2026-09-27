import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

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


class SequentialMultiTokenPredictionHead(nn.Module):
    """
    Sequential Multi-Token Prediction (MTP) modules in the DeepSeek-V3 style.

    Unlike ``MultiTokenPredictionHead``, each depth runs a full Transformer
    block supplied by the model and reuses the main model's output matrix, and
    depth ``k`` consumes the hidden state of depth ``k - 1``, which keeps the
    causal chain between the predictions.

    Structure (depth ``k = 1 .. num_heads`` at position ``i``, each depth with
    its own norms, projection and block):
        - ``h = RMSNorm(h_{k-1}[i])`` and ``e = RMSNorm(embedding[i + k])``
        - ``h_k[i] = block(Linear(concat(h, e)))``, with a bias-free
          ``2 * embed_dim -> embed_dim`` projection
        - ``logits_k[i] = RMSNorm(h_k[i]) @ output_weight.T`` predicts token
          ``i + k + 1``. The output norm plays the role of the main model's
          final norm (``shared_head.norm`` in DeepSeek-V3 checkpoints); the
          next depth reads the un-normalized ``h_k``.

    Args:
        embed_dim (int): Model hidden dimension.
        output_weight (nn.Parameter): The main language-model output matrix,
            ``[vocab_size, embed_dim]``. The same parameter is reused, not
            copied.
        block_factory (Callable[[], nn.Module]): Builds one Transformer block
            per depth. Each block maps ``[batch, seq, embed_dim]`` to the same
            shape. The head applies its own output RMSNorm, so the block
            should not end in a final norm.
        num_heads (int, optional): Number of extra future-token depths.
            Defaults to 1.
        rms_norm_eps (float, optional): Epsilon of the three RMSNorms at each
            depth (hidden input, embedding input and output). Defaults to 1e-6.

    Forward:
        Takes the main model's final hidden states and the token embeddings,
        both ``[batch, seq_len, embed_dim]``. Returns a list with one logits
        tensor per depth; depth ``k`` has shape
        ``[batch, seq_len - k, vocab_size]``. Depths with no positions left
        are skipped.
    """

    def __init__(
        self,
        embed_dim: int,
        output_weight: nn.Parameter,
        block_factory: Callable[[], nn.Module],
        num_heads: int = 1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if output_weight.ndim != 2 or output_weight.shape[1] != embed_dim:
            raise ValueError("output_weight must have shape [vocab_size, embed_dim]")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_weight = output_weight
        self.hidden_norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=rms_norm_eps) for _ in range(num_heads)]
        )
        self.embedding_norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=rms_norm_eps) for _ in range(num_heads)]
        )
        self.projections = nn.ModuleList(
            [Linear(2 * embed_dim, embed_dim, bias=False) for _ in range(num_heads)]
        )
        self.blocks = nn.ModuleList([block_factory() for _ in range(num_heads)])
        self.output_norms = nn.ModuleList(
            [RMSNorm(embed_dim, eps=rms_norm_eps) for _ in range(num_heads)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Run the prediction depths in order.

        Args:
            hidden_states (torch.Tensor): Hidden states from the main model
                (``h_0``), ``[batch, seq_len, embed_dim]``.
            token_embeddings (torch.Tensor): Input token embeddings, same shape.

        Returns:
            list[torch.Tensor]: Logits per depth,
            ``[batch, seq_len - k, vocab_size]`` for depth ``k``.
        """
        if hidden_states.shape != token_embeddings.shape:
            raise ValueError("hidden_states and token_embeddings must have equal shape")

        sequence_length = hidden_states.shape[1]
        previous_hidden = hidden_states
        predictions = []

        for depth in range(self.num_heads):
            offset = depth + 1
            target_length = sequence_length - offset
            if target_length < 1:
                break

            hidden = self.hidden_norms[depth](previous_hidden[:, :target_length])
            embedding = self.embedding_norms[depth](
                token_embeddings[:, offset : offset + target_length]
            )
            combined = torch.cat([hidden, embedding], dim=-1)
            previous_hidden = self.blocks[depth](self.projections[depth](combined))
            output = self.output_norms[depth](previous_hidden)
            predictions.append(F.linear(output, self.output_weight))

        return predictions
