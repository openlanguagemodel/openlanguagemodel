from __future__ import annotations

from typing import Callable, Literal, Type

import torch
import torch.nn as nn

from olm.nn.attention import GroupedQueryAttention, SlidingWindowAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.moe import MoEFeedForward
from olm.nn.norms import RMSNorm


LayerKind = Literal["dense", "moe"]
AttentionKind = Literal["local", "global"]


def dense_first_moe_schedule(num_layers: int, first_dense_layers: int) -> list[LayerKind]:
    """Return ``dense`` for the first N layers, then ``moe``."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if first_dense_layers < 0:
        raise ValueError("first_dense_layers must be >= 0")

    return ["dense" if i < first_dense_layers else "moe" for i in range(num_layers)]


def local_global_attention_schedule(
    num_layers: int,
    local_layers: int,
    global_layers: int = 1,
    start_with_global: bool = False,
) -> list[AttentionKind]:
    """
    Build repeating local/global attention schedules.

    Examples:
        ``local_layers=3, global_layers=1`` gives a 3:1 local/global pattern.
        ``local_layers=5, global_layers=1`` gives a 5:1 local/global pattern.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if local_layers < 0 or global_layers < 0:
        raise ValueError("local_layers and global_layers must be >= 0")
    if local_layers + global_layers == 0:
        raise ValueError("at least one local or global layer is required")

    local = ["local"] * local_layers
    global_ = ["global"] * global_layers
    pattern = (global_ + local) if start_with_global else (local + global_)

    schedule: list[AttentionKind] = []
    while len(schedule) < num_layers:
        schedule.extend(pattern)
    return schedule[:num_layers]


class PreNormTransformerBlock(nn.Module):
    """Pre-norm residual attention + feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        attention: nn.Module,
        feed_forward: nn.Module,
        norm_cls: Type[nn.Module] = RMSNorm,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = norm_cls(embed_dim, eps=norm_eps)
        self.attention = attention
        self.ffn_norm = norm_cls(embed_dim, eps=norm_eps)
        self.feed_forward = feed_forward
        self.last_router_logits = None
        self.last_router_stats = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.attention(self.attn_norm(x), **kwargs)
        x = x + h

        h = self.feed_forward(self.ffn_norm(x))
        if isinstance(h, tuple):
            h, router_logits = h
            self.last_router_logits = router_logits
            if hasattr(self.feed_forward, "get_router_stats"):
                self.last_router_stats = self.feed_forward.get_router_stats()

        return x + h


class ParallelTransformerBlock(nn.Module):
    """Parallel attention and feed-forward residual block."""

    def __init__(
        self,
        embed_dim: int,
        attention: nn.Module,
        feed_forward: nn.Module,
        norm_cls: Type[nn.Module] = RMSNorm,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = norm_cls(embed_dim, eps=norm_eps)
        self.ffn_norm = norm_cls(embed_dim, eps=norm_eps)
        self.attention = attention
        self.feed_forward = feed_forward
        self.last_router_logits = None
        self.last_router_stats = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        attn_out = self.attention(self.attn_norm(x), **kwargs)
        ffn_out = self.feed_forward(self.ffn_norm(x))
        if isinstance(ffn_out, tuple):
            ffn_out, router_logits = ffn_out
            self.last_router_logits = router_logits
            if hasattr(self.feed_forward, "get_router_stats"):
                self.last_router_stats = self.feed_forward.get_router_stats()
        return x + attn_out + ffn_out


def build_dense_transformer_block(
    embed_dim: int,
    attention: nn.Module,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    bias: bool = False,
    norm_eps: float = 1e-6,
    parallel: bool = False,
) -> nn.Module:
    ffn = SwiGLUFFN(
        embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        bias=bias,
    )
    block_cls = ParallelTransformerBlock if parallel else PreNormTransformerBlock
    return block_cls(embed_dim, attention, ffn, norm_eps=norm_eps)


def build_moe_transformer_block(
    embed_dim: int,
    attention: nn.Module,
    num_experts: int,
    top_k: int,
    expert_hidden_dim: int,
    num_shared_experts: int = 0,
    dropout: float = 0.0,
    bias: bool = False,
    scoring_func: Literal["softmax", "sigmoid"] = "softmax",
    routing_method: Literal["topk", "noaux_tc"] = "topk",
    use_router_bias: bool = False,
    fp32_gate: bool = False,
    routed_scaling_factor: float = 1.0,
    norm_eps: float = 1e-6,
    parallel: bool = False,
) -> nn.Module:
    moe = MoEFeedForward(
        embed_dim=embed_dim,
        expert_cls=SwiGLUFFN,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        expert_kwargs={
            "hidden_dim": expert_hidden_dim,
            "dropout": dropout,
            "bias": bias,
        },
        scoring_func=scoring_func,
        routing_method=routing_method,
        use_router_bias=use_router_bias,
        fp32_gate=fp32_gate,
        routed_scaling_factor=routed_scaling_factor,
    )
    block_cls = ParallelTransformerBlock if parallel else PreNormTransformerBlock
    return block_cls(embed_dim, attention, moe, norm_eps=norm_eps)


def build_local_swa_block(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    window_size: int,
    hidden_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    attention = SlidingWindowAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        window_size=window_size,
        head_dim=kwargs.pop("head_dim", None),
        rope_theta=kwargs.pop("rope_theta", 10000.0),
        use_qk_norm=kwargs.pop("use_qk_norm", False),
        partial_rotary_factor=kwargs.pop("partial_rotary_factor", 1.0),
        use_attention_sink=kwargs.pop("use_attention_sink", False),
    )
    return build_dense_transformer_block(
        embed_dim,
        attention,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def build_global_nope_block(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    hidden_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    attention = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=kwargs.pop("head_dim", None),
        use_rope=False,
        use_qk_norm=kwargs.pop("use_qk_norm", False),
        qkv_bias=kwargs.pop("qkv_bias", False),
        use_bias=kwargs.pop("use_bias", False),
    )
    return build_dense_transformer_block(
        embed_dim,
        attention,
        hidden_dim=hidden_dim,
        **kwargs,
    )
