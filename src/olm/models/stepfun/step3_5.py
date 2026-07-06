import torch
import torch.nn as nn
from typing import List

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention, SlidingWindowAttention
from olm.nn.attention.head_gated import HeadGate
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead, MultiTokenPredictionHead
from olm.nn.moe import MoEFeedForward


class Step3_5_FullAttnBlock(Block):
    """
    Full-attention block for Step 3.5 Flash.

    GQA with 64 heads, 8 KV groups, per-head sigmoid gating, QK-Norm,
    and partial RoPE (rotary_factor=0.5).  Uses a higher RoPE theta
    (5M) with Llama3-style frequency scaling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rms_norm_eps: float,
        ffn_module: nn.Module,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads, max_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_bias=False,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
        )
        self.head_gate = HeadGate(embed_dim, num_heads)
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = ffn_module
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        h = self.attn_norm(x)
        attn_out = self.self_attn(h, **{k: v for k, v in kwargs.items() if k == "mask"})
        B, N, _ = attn_out.shape
        num_heads = self.head_gate.gate_proj.shape[0]
        head_dim = attn_out.shape[-1] // num_heads
        attn_out_heads = attn_out.view(B, N, num_heads, head_dim)
        attn_out_heads = self.head_gate(x, attn_out_heads)
        attn_out = attn_out_heads.view(B, N, -1)
        x = residual + attn_out

        residual = x
        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoEFeedForward):
            h, router_logits = self.ffn(h)
            self.last_router_logits = router_logits
        else:
            h = self.ffn(h)
        return residual + h


class Step3_5_SlidingAttnBlock(Block):
    """
    Sliding-window attention block for Step 3.5 Flash.

    GQA with 96 heads, 8 KV groups, window_size=512, per-head sigmoid
    gating, QK-Norm, and full RoPE (rotary_factor=1.0) at theta=10K.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        window_size: int,
        rope_theta: float,
        rms_norm_eps: float,
        ffn_module: nn.Module,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = SlidingWindowAttention(
            embed_dim, num_heads, num_kv_heads, max_seq_len,
            window_size=window_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
        )
        self.head_gate = HeadGate(embed_dim, num_heads)
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = ffn_module
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        h = self.attn_norm(x)
        attn_out = self.self_attn(h, **{k: v for k, v in kwargs.items() if k == "mask"})
        B, N, _ = attn_out.shape
        num_heads = self.head_gate.gate_proj.shape[0]
        head_dim = attn_out.shape[-1] // num_heads
        attn_out_heads = attn_out.view(B, N, num_heads, head_dim)
        attn_out_heads = self.head_gate(x, attn_out_heads)
        attn_out = attn_out_heads.view(B, N, -1)
        x = residual + attn_out

        residual = x
        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoEFeedForward):
            h, router_logits = self.ffn(h)
            self.last_router_logits = router_logits
        else:
            h = self.ffn(h)
        return residual + h


STEP3_5_LAYER_TYPES = [
    "full", "sliding", "sliding", "sliding",
] * 12


STEP3_5_MOE_LAYERS = set(range(3, 45))

STEP3_5_ROPE_THETA = [
    5000000.0, 10000.0, 10000.0, 10000.0,
] * 12

STEP3_5_PARTIAL_ROTARY = [
    0.5, 1.0, 1.0, 1.0,
] * 12


class Step3_5_Flash_Model(Block):
    """
    Step 3.5 Flash -- throughput-oriented 196B sparse MoE.

    48 layers in a 4:1 sliding-window / full-attention pattern.  Full
    attention layers use 64 heads with partial RoPE (0.5) at theta=5M;
    sliding-window layers use 96 heads with full RoPE at theta=10K.
    All layers have per-head sigmoid gating and QK-Norm.

    42 of 48 layers (indices 3--44) use a 288-expert sigmoid-routed MoE
    with top-8 selection, router bias, FP32 gate, weight normalization,
    and 3x routed scaling.  Layers 0--2 and 45--47 use dense SwiGLU.
    Three MTP heads for speculative decoding.

    Structure:
        Embedding -> [FullAttn | SlidingAttn] x 48 -> RMSNorm -> OutputHead
        + MultiTokenPredictionHead (3 heads)

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 128896,
        embed_dim: int = 4096,
        intermediate_size: int = 11264,
        moe_intermediate_size: int = 1280,
        num_layers: int = 48,
        full_attn_heads: int = 64,
        sliding_attn_heads: int = 96,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 262144,
        window_size: int = 512,
        num_experts: int = 288,
        num_experts_per_tok: int = 8,
        num_mtp_heads: int = 3,
        routed_scaling_factor: float = 3.0,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = False,
        layer_types: List[str] = None,
        moe_layers: set = None,
        rope_thetas: List[float] = None,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layer_types = layer_types or STEP3_5_LAYER_TYPES[:num_layers]
        moe_layers = moe_layers if moe_layers is not None else STEP3_5_MOE_LAYERS
        rope_thetas = rope_thetas or STEP3_5_ROPE_THETA[:num_layers]

        layers = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_layers:
                ffn = MoEFeedForward(
                    embed_dim=embed_dim,
                    expert_cls=SwiGLUFFN,
                    num_experts=num_experts,
                    num_shared_experts=1,
                    top_k=num_experts_per_tok,
                    expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
                    scoring_func="sigmoid",
                    use_router_bias=True,
                    norm_weights=True,
                    fp32_gate=True,
                    routed_scaling_factor=routed_scaling_factor,
                )
            else:
                ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

            if layer_types[i] == "full":
                layers.append(Step3_5_FullAttnBlock(
                    embed_dim, full_attn_heads, num_kv_heads, head_dim,
                    max_seq_len, rope_thetas[i], rms_norm_eps, ffn,
                ))
            else:
                layers.append(Step3_5_SlidingAttnBlock(
                    embed_dim, sliding_attn_heads, num_kv_heads, head_dim,
                    max_seq_len, window_size, rope_thetas[i], rms_norm_eps, ffn,
                ))

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim, vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights, use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers
        self.mtp = MultiTokenPredictionHead(
            embed_dim, vocab_size, num_heads=num_mtp_heads,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        token_embeddings = self.blocks[0](x)

        hidden = token_embeddings
        for block in self.transformer_blocks:
            hidden = block(hidden, **kwargs)

        hidden = self.blocks[1](hidden)
        return self.blocks[2](hidden)

    def forward_with_mtp(self, x: torch.Tensor, **kwargs):
        """Return main logits and MTP auxiliary logits for training."""
        token_embeddings = self.blocks[0](x)

        hidden = token_embeddings
        for block in self.transformer_blocks:
            hidden = block(hidden, **kwargs)

        hidden = self.blocks[1](hidden)
        logits = self.blocks[2](hidden)
        mtp_logits = self.mtp(hidden, token_embeddings)
        return logits, mtp_logits


class Step3_5_Flash_196B(Step3_5_Flash_Model):
    """Step 3.5 Flash 196B with default config."""

    def __init__(self):
        super().__init__()
