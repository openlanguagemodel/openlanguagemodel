# src/olm/nn/attention/dsa.py
"""
DeepSeek Sparse Attention (DSA), introduced with DeepSeek-V3.2.

DSA keeps Multi-head Latent Attention (MLA) and adds a small *lightning
indexer* in front of it. For every query token ``t`` the indexer scores each
preceding token ``s``:

    I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])

with a handful of indexer heads ``j``. Only the ``index_topk`` highest-scoring
keys are kept, and the main MLA heads attend over that subset.

DSA is added to an already-trained dense model in two continued-training
stages, both supported by ``DeepSeekSparseAttention.forward``:

1. Dense warm-up (``use_sparse_attention=False``): attention stays dense and
   only the indexer learns (freezing the other parameters is up to the
   training loop), by minimizing the KL divergence between the main attention
   distribution, summed over heads and renormalized, and the softmax of the
   indexer scores.
2. Sparse training (``use_sparse_attention=True``): the top-k selection is
   applied and the KL loss is restricted to the selected keys. The indexer
   input is detached, so the indexer learns only from its KL loss and the rest
   of the model only from the language-modeling loss.

Pass ``compute_alignment_loss=True`` to get that KL loss in
``module.last_stats.alignment_loss``; adding it to the training loss is up to
the caller.

This is a readable reference, not a fast kernel. The selection is applied as
a boolean mask to PyTorch SDPA, so the result matches sparse attention but
the cost is still ``O(L^2)``, not the ``O(L * k)`` of fused sparse kernels.
The indexer runs in FP32; DeepSeek notes it can run in FP8 for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.attention.mla import MultiHeadLatentAttention, _apply_rope_layout
from olm.nn.embeddings.positional.rope import (
    RotaryPositionalEmbedding,
    ScaledRotaryPositionalEmbedding,
)
from olm.nn.torch_nn_wrappers import Linear


@dataclass
class DeepSeekSparseAttentionStats:
    """
    Token-selection metadata from the most recent DSA forward pass.

    Attributes:
        index_scores (torch.Tensor): Lightning-indexer scores
            ``[batch, seq_len, seq_len]``; masked positions are ``-inf``.
        selected_indices (torch.Tensor): Top-k key positions per query
            ``[batch, seq_len, min(index_topk, seq_len)]``.
        selection_mask (torch.Tensor): Boolean ``[batch, seq_len, seq_len]``
            mask of the selected (and valid) keys.
        metadata (dict): Configuration of the pass (``index_topk``, ranks,
            ``use_sparse_attention``).
        target_distribution (torch.Tensor, optional): Main attention
            distribution the indexer is aligned to, when requested.
        alignment_loss (torch.Tensor, optional): Indexer KL loss, when
            requested.
    """

    index_scores: torch.Tensor
    selected_indices: torch.Tensor
    selection_mask: torch.Tensor
    metadata: dict
    target_distribution: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None


def _make_rope(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    scaling_type: Optional[str],
    scaling_factor: float,
    original_max_seq_len: Optional[int],
    yarn_beta_fast: float,
    yarn_beta_slow: float,
) -> nn.Module:
    """Build plain RoPE, or scaled RoPE when ``scaling_type`` is set."""
    if scaling_type is None:
        return RotaryPositionalEmbedding(
            head_dim, max_seq_len=max_seq_len, base=rope_theta
        )
    return ScaledRotaryPositionalEmbedding(
        head_dim,
        max_seq_len=max_seq_len,
        base=rope_theta,
        scaling_type=scaling_type,
        scaling_factor=scaling_factor,
        original_max_seq_len=original_max_seq_len,
        yarn_beta=yarn_beta_fast,
        yarn_beta_slow=yarn_beta_slow,
    )


class LightningIndexer(nn.Module):
    """
    Lightning indexer: the cheap token scorer of DeepSeek Sparse Attention.

    Structure:
        - Queries: ``q_proj`` maps the query input (MLA's normalized query
          latent inside DSA, or ``x``) to ``num_heads`` heads of ``head_dim``.
        - Keys: one shared key per token, ``k_norm(k_proj(x))`` (a LayerNorm,
          as in the reference implementation).
        - The first ``rotary_dim`` channels of queries and keys get RoPE.
        - Head weights: ``w_proj(x) * num_heads ** -0.5``, computed in FP32.
        - Score: ``sum_j w[t, j] * ReLU(q[t, j] . k[s] * head_dim ** -0.5)``.

    Args:
        embed_dim (int): Model hidden dimension (input to keys and weights).
        num_heads (int): Number of indexer heads.
        head_dim (int): Per-head indexer dimension.
        query_dim (int, optional): Width of the query input. Defaults to
            ``embed_dim``.
        rotary_dim (int, optional): Number of leading channels that get RoPE.
            Must be even and at most ``head_dim``. Defaults to 0 (no RoPE).
        max_seq_len (int, optional): Maximum sequence length for RoPE.
            Defaults to 2048.
        rope_theta (float, optional): RoPE base frequency. Defaults to 10000.0.
        rope_scaling_type (str, optional): ``ScaledRotaryPositionalEmbedding``
            scaling type (e.g. ``"yarn"``). Defaults to None (plain RoPE).
        rope_scaling_factor (float, optional): Context-extension factor.
            Defaults to 1.0.
        original_max_seq_len (int, optional): Pre-extension context length for
            scaled RoPE. Defaults to ``max_seq_len``.
        yarn_beta_fast (float, optional): YaRN ``beta_fast``. Defaults to 32.0.
        yarn_beta_slow (float, optional): YaRN ``beta_slow``. Defaults to 1.0.
        rope_interleaved (bool, optional): RoPE dimension layout. True pairs
            adjacent channels, False pairs channel ``i`` with
            ``i + rotary_dim / 2``. Defaults to False.
        bias (bool, optional): Bias on the indexer projections. Defaults to
            False.
        norm_eps (float, optional): Epsilon of the key LayerNorm. Defaults to
            1e-6.

    Forward:
        Takes ``x`` ``[batch, seq_len, embed_dim]``, optional ``query_states``
        ``[batch, seq_len, query_dim]`` (defaults to ``x``) and an optional
        boolean/binary/additive ``mask``. Returns float32 scores
        ``[batch, seq_len, seq_len]`` with ``-inf`` at future and masked keys.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        query_dim: Optional[int] = None,
        rotary_dim: int = 0,
        max_seq_len: int = 2048,
        rope_theta: float = 10_000.0,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        rope_interleaved: bool = False,
        bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if rotary_dim < 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(
                f"rotary_dim must be even and in [0, head_dim={head_dim}], "
                f"got {rotary_dim}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_interleaved = rope_interleaved
        self.scale = head_dim**-0.5

        self.q_proj = Linear(query_dim or embed_dim, num_heads * head_dim, bias=bias)
        self.k_proj = Linear(embed_dim, head_dim, bias=bias)
        self.k_norm = nn.LayerNorm(head_dim, eps=norm_eps)
        self.w_proj = Linear(embed_dim, num_heads, bias=bias)
        self.rope = (
            _make_rope(
                rotary_dim,
                max_seq_len,
                rope_theta,
                rope_scaling_type,
                rope_scaling_factor,
                original_max_seq_len,
                yarn_beta_fast,
                yarn_beta_slow,
            )
            if rotary_dim
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        query_states: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score every (query, key) pair.

        Args:
            x (torch.Tensor): Hidden states ``[batch, seq_len, embed_dim]``.
            query_states (torch.Tensor, optional): Query input
                ``[batch, seq_len, query_dim]``. Defaults to ``x``.
            mask (torch.Tensor, optional): Attention mask broadcastable to
                ``[batch, seq_len, seq_len]`` or ``[batch, heads, seq_len,
                seq_len]`` (a key counts as masked if any head masks it).
                Defaults to None.

        Returns:
            torch.Tensor: Float32 scores ``[batch, seq_len, seq_len]``.
        """
        batch, seq_len, _ = x.shape
        query_states = x if query_states is None else query_states
        q = self.q_proj(query_states).view(
            batch, seq_len, self.num_heads, self.head_dim
        )
        # One key per token, shared by all indexer heads: [B, S, head_dim]
        k = self.k_norm(self.k_proj(x))

        if self.rope is not None:
            q_rot, q_pass = torch.split(
                q, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1
            )
            k_rot, k_pass = torch.split(
                k, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1
            )
            q_rot = _apply_rope_layout(self.rope, q_rot, self.rope_interleaved)
            k_rot = _apply_rope_layout(
                self.rope, k_rot.unsqueeze(2), self.rope_interleaved
            ).squeeze(2)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)

        # Per-head rectified similarities: [B, H, T, S]
        per_head = torch.einsum("bthd,bsd->bhts", q.float(), k.float())
        per_head = F.relu(per_head * self.scale)

        # Query-conditioned head weights, kept in FP32 under autocast: [B, T, H]
        bias = self.w_proj.bias
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = F.linear(
                x.float(),
                self.w_proj.weight.float(),
                None if bias is None else bias.float(),
            )
        weights = weights * (self.num_heads**-0.5)
        scores = (per_head * weights.transpose(1, 2).unsqueeze(-1)).sum(dim=1)

        valid = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        if mask is not None:
            user_mask = attention_mask_to_bool(mask, device=x.device)
            if user_mask.dim() == 4:
                user_mask = user_mask.all(dim=1)
            valid = valid & user_mask
        return scores.masked_fill(~valid, float("-inf"))


def causal_topk_selection(
    index_scores: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select up to ``top_k`` keys per query from indexer scores.

    Keys whose score is ``-inf`` (future or masked positions) are never marked
    as selected, so early queries with fewer than ``top_k`` valid keys select
    only those.

    Args:
        index_scores (torch.Tensor): Scores ``[batch, seq_len, seq_len]``.
        top_k (int): Keys to keep per query. Must be >= 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(selected_indices,
        selection_mask)``: indices ``[batch, seq_len, min(top_k, seq_len)]``
        and a boolean mask ``[batch, seq_len, seq_len]``.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if index_scores.dim() != 3 or index_scores.shape[-2] != index_scores.shape[-1]:
        raise ValueError("index_scores must have shape [batch, seq_len, seq_len]")

    batch, seq_len, _ = index_scores.shape
    k_eff = min(top_k, seq_len)
    selected_scores, selected_indices = torch.topk(index_scores, k_eff, dim=-1)
    selected_valid = torch.isfinite(selected_scores)
    selection_mask = torch.zeros(
        batch, seq_len, seq_len, device=index_scores.device, dtype=torch.bool
    )
    selection_mask.scatter_(-1, selected_indices, selected_valid)
    return selected_indices, selection_mask


def indexer_kl_alignment_loss(
    index_scores: torch.Tensor,
    target_scores: torch.Tensor,
    selection_mask: Optional[torch.Tensor] = None,
    target_is_probabilities: bool = False,
) -> torch.Tensor:
    """
    KL divergence that aligns the lightning indexer with the main attention.

    Computes ``KL(p || softmax(index_scores))`` per query over the valid keys
    and averages over queries. ``p`` is ``softmax(target_scores)``, or
    ``target_scores`` renormalized over the valid keys when
    ``target_is_probabilities`` is True.

    Args:
        index_scores (torch.Tensor): Indexer scores ``[batch, seq_len,
            seq_len]``; ``-inf`` marks invalid keys.
        target_scores (torch.Tensor): Target logits or probabilities, same
            shape as ``index_scores``.
        selection_mask (torch.Tensor, optional): Boolean mask restricting the
            loss to selected keys (the sparse training stage). Defaults to
            None (all finite keys, the dense warm-up stage).
        target_is_probabilities (bool, optional): Whether ``target_scores`` is
            already a distribution. Defaults to False.

    Returns:
        torch.Tensor: Scalar loss.
    """
    if index_scores.shape != target_scores.shape:
        raise ValueError("index_scores and target_scores must have the same shape")

    valid = torch.isfinite(index_scores)
    if selection_mask is not None:
        if selection_mask.shape != index_scores.shape:
            raise ValueError("selection_mask must match index_scores")
        valid &= selection_mask
    if not valid.any(dim=-1).all():
        raise ValueError("every query row must contain at least one valid key")

    masked_index = index_scores.masked_fill(~valid, float("-inf"))
    if target_is_probabilities:
        target = target_scores.masked_fill(~valid, 0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        masked_target = target_scores.masked_fill(~valid, float("-inf"))
        target = F.softmax(masked_target, dim=-1)

    log_index = F.log_softmax(masked_index, dim=-1)
    target_log = target.clamp_min(1e-12).log()
    loss = target * (target_log - log_index)
    return loss.masked_fill(~valid, 0.0).sum(dim=-1).mean()


class DeepSeekSparseAttention(MultiHeadLatentAttention):
    """
    DeepSeek Sparse Attention (DSA): MLA plus a lightning indexer that picks
    ``index_topk`` keys per query (DeepSeek-V3.2).

    The MLA projections, decoupled RoPE and output projection are inherited
    unchanged from ``MultiHeadLatentAttention``. The indexer reads MLA's
    normalized query latent (``q_lora_rank`` wide) and the hidden states; its
    top-k selection becomes a boolean attention mask. See the module docstring
    for the two training stages.

    Args:
        embed_dim (int): Model hidden dimension.
        num_heads (int): Number of MLA attention heads.
        max_seq_len (int): Maximum sequence length for RoPE.
        kv_lora_rank (int): Rank of the compressed key/value latent.
        qk_nope_head_dim (int): Per-head query/key dimension without RoPE.
        qk_rope_head_dim (int): Per-head query/key dimension with RoPE. The
            indexer rotates the same number of leading channels.
        v_head_dim (int): Per-head value dimension.
        index_n_heads (int): Number of lightning-indexer heads.
        index_head_dim (int): Per-head lightning-indexer dimension.
        index_topk (int): Keys kept per query in the sparse stage.
        q_lora_rank (int, optional): Rank of the query latent. None projects
            queries directly from ``x``. Defaults to None.
        dropout (float, optional): Attention dropout. Defaults to 0.0.
        rope_theta (float, optional): RoPE base frequency. Defaults to 10000.0.
        rope_scaling_type (str, optional): ``ScaledRotaryPositionalEmbedding``
            scaling type for both MLA and the indexer (e.g. ``"yarn"``).
            Defaults to None (plain RoPE).
        rope_scaling_factor (float, optional): Context-extension factor.
            Defaults to 1.0.
        original_max_seq_len (int, optional): Pre-extension context length.
            Defaults to ``max_seq_len``.
        yarn_beta_fast (float, optional): YaRN ``beta_fast``. Defaults to 32.0.
        yarn_beta_slow (float, optional): YaRN ``beta_slow``. Defaults to 1.0.
        yarn_mscale_all_dim (float, optional): YaRN ``mscale_all_dim``. When
            non-zero and ``rope_scaling_type="yarn"``, the softmax scale is
            multiplied by ``(0.1 * mscale_all_dim * ln(factor) + 1) ** 2``
            (DeepSeek-V3's YaRN correction). Defaults to 0.0 (no change).
        rope_interleaved (bool, optional): MLA RoPE layout (see
            ``MultiHeadLatentAttention``). Defaults to True.
        indexer_rope_interleaved (bool, optional): Indexer RoPE layout.
            Defaults to False.
        bias (bool, optional): Bias on all projections. Defaults to False.
        rms_norm_eps (float, optional): Epsilon of the MLA latent RMSNorms and
            the indexer key LayerNorm. Defaults to 1e-6.
        detach_indexer_input (bool, optional): Detach the indexer's inputs so
            that its KL loss does not reach the rest of the model. Defaults to
            True.

    Forward:
        Takes ``x`` ``[batch, seq_len, embed_dim]`` and an optional ``mask``
        (True or 1 means attend). ``use_sparse_attention`` picks the training
        stage and ``compute_alignment_loss`` adds the indexer KL loss to
        ``last_stats``. Returns ``[batch, seq_len, embed_dim]``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        q_lora_rank: Optional[int] = None,
        dropout: float = 0.0,
        rope_theta: float = 10_000.0,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        original_max_seq_len: Optional[int] = None,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale_all_dim: float = 0.0,
        rope_interleaved: bool = True,
        indexer_rope_interleaved: bool = False,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
        detach_indexer_input: bool = True,
    ):
        if index_topk < 1:
            raise ValueError(f"index_topk must be >= 1, got {index_topk}")

        attention_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        if rope_scaling_type == "yarn" and yarn_mscale_all_dim:
            mscale = (
                1.0
                if rope_scaling_factor <= 1
                else 0.1 * yarn_mscale_all_dim * math.log(rope_scaling_factor) + 1.0
            )
            attention_scale *= mscale * mscale

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            dropout=dropout,
            rope_theta=rope_theta,
            bias=bias,
            rms_norm_eps=rms_norm_eps,
            attention_scale=attention_scale,
            rope_interleaved=rope_interleaved,
        )
        self.index_topk = index_topk
        self.detach_indexer_input = detach_indexer_input
        # Replace MLA's plain RoPE so that context extension (e.g. YaRN) applies.
        self.rope = _make_rope(
            qk_rope_head_dim,
            max_seq_len,
            rope_theta,
            rope_scaling_type,
            rope_scaling_factor,
            original_max_seq_len,
            yarn_beta_fast,
            yarn_beta_slow,
        )
        self.indexer = LightningIndexer(
            embed_dim=embed_dim,
            query_dim=q_lora_rank or embed_dim,
            num_heads=index_n_heads,
            head_dim=index_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            original_max_seq_len=original_max_seq_len,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            rope_interleaved=indexer_rope_interleaved,
            bias=bias,
            norm_eps=rms_norm_eps,
        )
        self.last_stats: DeepSeekSparseAttentionStats | None = None

    def _main_attention_distribution(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Dense MLA attention probabilities summed over heads and L1-normalized,
        the indexer's KL target. Returns ``[batch, seq_len, seq_len]``.
        """
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        seq_len = q.shape[-2]
        valid = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
        if mask is not None:
            valid = valid & mask
        scores = scores.masked_fill(~valid, float("-inf"))
        target = F.softmax(scores, dim=-1).sum(dim=1)
        return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        compute_alignment_loss: bool = False,
        alignment_on_selected: bool = True,
        use_sparse_attention: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for either DSA training stage.

        Args:
            x (torch.Tensor): Input ``[batch, seq_len, embed_dim]``.
            mask (torch.Tensor, optional): Attention mask broadcastable to
                ``[batch, heads, seq_len, seq_len]``; a 3-D mask is read as
                ``[batch, seq_len, seq_len]``. Defaults to None.
            compute_alignment_loss (bool, optional): Compute the indexer KL
                loss into ``last_stats.alignment_loss``. Defaults to False.
            alignment_on_selected (bool, optional): In the sparse stage,
                restrict the KL loss to the selected keys. Defaults to True.
            use_sparse_attention (bool, optional): False keeps dense causal
                attention (indexer warm-up); True attends only to the
                ``index_topk`` selected keys. Defaults to True.

        Returns:
            torch.Tensor: Output ``[batch, seq_len, embed_dim]``.
        """
        batch, seq_len, _ = x.shape
        if mask is not None:
            mask = attention_mask_to_bool(mask, device=x.device)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        q, k, value, q_latent = self._project_qkv(x)

        # --- Lightning indexer and top-k selection ---
        index_x = x.detach() if self.detach_indexer_input else x
        index_q = q_latent.detach() if self.detach_indexer_input else q_latent
        index_scores = self.indexer(index_x, query_states=index_q, mask=mask)
        selected_indices, selection_mask = causal_topk_selection(
            index_scores, self.index_topk
        )

        # --- Main attention: dense (warm-up) or over the selected keys ---
        if use_sparse_attention:
            attention_mask = selection_mask.unsqueeze(1)
            if mask is not None:
                attention_mask = attention_mask & mask
        else:
            attention_mask = mask
        attention_out = self.compute_attention(q, k, value, attention_mask)
        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.v_head_dim)
        )

        # --- Indexer alignment loss ---
        target_distribution = None
        alignment_loss = None
        if compute_alignment_loss:
            target_distribution = self._main_attention_distribution(
                q.detach(), k.detach(), mask
            )
            alignment_loss = indexer_kl_alignment_loss(
                index_scores,
                target_distribution,
                selection_mask=(
                    selection_mask
                    if alignment_on_selected and use_sparse_attention
                    else None
                ),
                target_is_probabilities=True,
            )

        self.last_stats = DeepSeekSparseAttentionStats(
            index_scores=index_scores,
            selected_indices=selected_indices,
            selection_mask=selection_mask,
            target_distribution=target_distribution,
            alignment_loss=alignment_loss,
            metadata={
                "index_topk": self.index_topk,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "mode": "reference_masked_sdpa",
                "use_sparse_attention": use_sparse_attention,
            },
        )
        return self.out_proj(attention_out)
