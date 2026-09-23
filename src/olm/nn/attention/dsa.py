from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olm.nn.attention.masks import attention_mask_to_bool
from olm.nn.attention.mla import MultiHeadLatentAttention
from olm.nn.embeddings.positional.rope import (
    RotaryPositionalEmbedding,
    ScaledRotaryPositionalEmbedding,
)
from olm.nn.torch_nn_wrappers import Linear


@dataclass
class DeepSeekSparseAttentionStats:
    """Token-selection metadata from the most recent DSA forward pass."""

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


def _apply_rope_layout(
    rope: nn.Module,
    x: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        return rope(x)

    half = x.shape[-1] // 2
    paired = torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)
    paired = rope(paired)
    paired = paired.view(*paired.shape[:-1], half, 2)
    return torch.cat((paired[..., 0], paired[..., 1]), dim=-1)


class LightningIndexer(nn.Module):
    """Query-dependent token indexer used by DeepSeek Sparse Attention.

    The query projection consumes MLA's normalized query latent when one is
    supplied. Keys have an independent projection and LayerNorm. A small RoPE
    slice is applied before rectified per-head similarities are combined by
    query-conditioned head weights.
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
            raise ValueError("rotary_dim must be even and within [0, head_dim]")

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
        """Return causal index scores shaped ``[batch, query, key]``."""
        batch, seq_len, _ = x.shape
        query_states = x if query_states is None else query_states
        q = self.q_proj(query_states).view(
            batch, seq_len, self.num_heads, self.head_dim
        )
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

        per_head = torch.einsum("bthd,bsd->bhts", q.float(), k.float())
        per_head = F.relu(per_head * self.scale)
        bias = self.w_proj.bias
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = F.linear(
                x.float(),
                self.w_proj.weight.float(),
                None if bias is None else bias.float(),
            )
        weights = weights * (self.num_heads**-0.5)
        scores = (per_head * weights.transpose(1, 2).unsqueeze(-1)).sum(dim=1)

        causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        valid = causal
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
    """Select at most ``top_k`` finite causal keys for every query."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
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
    """Align indexer scores with the main attention distribution."""
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
    """DeepSeek Sparse Attention implemented over OLM's MLA projections.

    The lightning indexer selects a query-dependent subset of preceding token
    positions. This readable implementation applies the resulting mask through
    PyTorch SDPA: it is mathematically sparse and trainable, but does not claim
    the fused ``O(L * k)`` kernels used by the released frontier checkpoints.
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
        yarn_beta: Optional[float] = None,
        yarn_mscale_all_dim: float = 0.0,
        rope_interleaved: bool = False,
        indexer_rope_interleaved: bool = False,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
        detach_indexer_input: bool = True,
    ):
        if index_topk < 1:
            raise ValueError("index_topk must be >= 1")
        if yarn_beta is not None:
            yarn_beta_fast = yarn_beta

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
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        seq_len = q.shape[-2]
        valid = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
        if mask is not None:
            valid = valid & attention_mask_to_bool(mask, device=q.device)
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
        """Return MLA output for indexer warm-up or sparse adaptation.

        ``use_sparse_attention=False`` keeps dense causal attention while the
        lightning indexer learns from the full target distribution. Switching
        it to True applies the selected top-k mask and optionally aligns only
        over selected positions, matching DSA's two training stages.
        """
        batch, seq_len, _ = x.shape
        q, k, value, q_latent = self._project_qkv(x)
        index_x = x.detach() if self.detach_indexer_input else x
        index_q = q_latent.detach() if self.detach_indexer_input else q_latent
        index_scores = self.indexer(index_x, query_states=index_q, mask=mask)
        selected_indices, selection_mask = causal_topk_selection(
            index_scores, self.index_topk
        )

        sparse_mask = selection_mask.unsqueeze(1)
        if mask is not None:
            user_mask = attention_mask_to_bool(mask, device=x.device)
            if user_mask.dim() == 3:
                user_mask = user_mask.unsqueeze(1)
            sparse_mask &= user_mask

        attention_mask = sparse_mask if use_sparse_attention else mask
        attention_out = self.compute_attention(q, k, value, attention_mask)
        attention_out = (
            attention_out.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.num_heads * self.v_head_dim)
        )

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
