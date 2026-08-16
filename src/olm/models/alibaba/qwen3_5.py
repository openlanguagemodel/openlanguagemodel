import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import GatedAttention, GatedDeltaNet
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead, MultiTokenPredictionHead
from olm.nn.moe import MoEFeedForward


class Qwen3_5_GatedAttnBlock(Block):
    """
    Full-attention block for Qwen3.5.

    Uses GatedAttention (softmax + output sigmoid gate) with 32 heads,
    2 KV heads, head_dim=256, partial RoPE (factor=0.25), QK-Norm,
    and a 512-expert MoE with top-10.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rope_theta: float,
        partial_rotary_factor: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = GatedAttention(
            embed_dim, num_heads, num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            use_qk_norm=True,
            rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={
                "hidden_dim": moe_intermediate_size, "bias": False,
            },
        )
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x, router_logits = self.moe(self.ffn_norm(x))
        self.last_router_logits = router_logits
        return residual + x


class Qwen3_5_DeltaNetBlock(Block):
    """
    Linear-attention block for Qwen3.5.

    Uses Gated DeltaNet with causal Conv1d, 16 key heads, 64 value
    heads, and a 512-expert MoE with top-10.
    """

    def __init__(
        self,
        embed_dim: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = GatedDeltaNet(
            embed_dim,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_size=conv_kernel_size,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={
                "hidden_dim": moe_intermediate_size, "bias": False,
            },
        )
        self.last_router_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x

        residual = x
        x, router_logits = self.moe(self.ffn_norm(x))
        self.last_router_logits = router_logits
        return residual + x


QWEN3_5_LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
] * 15


class Qwen3_5_Model(Block):
    """
    Qwen3.5 397B -- sparse hybrid with Gated DeltaNet / Gated Attention.

    60 layers in a 3:1 linear / full-attention pattern (45 DeltaNet +
    15 Gated Attention).  Full-attention layers use GQA with output
    gating, partial RoPE (0.25), QK-Norm.  Linear layers use Gated
    DeltaNet with causal Conv1d (kernel=4).  All layers use 512-expert
    MoE with top-10, one shared expert.  One MTP head.

    Structure:
        Embedding -> [DeltaNet | GatedAttn] x 60 -> RMSNorm -> OutputHead
        + MultiTokenPredictionHead (1 head)

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 248320,
        embed_dim: int = 4096,
        num_layers: int = 60,
        full_attn_heads: int = 32,
        full_attn_kv_heads: int = 2,
        full_attn_head_dim: int = 256,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 64,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        max_seq_len: int = 262144,
        moe_intermediate_size: int = 1024,
        shared_expert_intermediate_size: int = 1024,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        num_mtp_heads: int = 1,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        rms_norm_eps: float = 1e-6,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)
        layer_types = QWEN3_5_LAYER_TYPES[:num_layers]

        layers = nn.ModuleList()
        for i in range(num_layers):
            if layer_types[i] == "full_attention":
                layers.append(Qwen3_5_GatedAttnBlock(
                    embed_dim, full_attn_heads, full_attn_kv_heads,
                    full_attn_head_dim, max_seq_len,
                    moe_intermediate_size, shared_expert_intermediate_size,
                    num_experts, num_experts_per_tok,
                    rope_theta, partial_rotary_factor, rms_norm_eps,
                ))
            else:
                layers.append(Qwen3_5_DeltaNetBlock(
                    embed_dim,
                    linear_num_key_heads, linear_num_value_heads,
                    linear_key_head_dim, linear_value_head_dim,
                    linear_conv_kernel_dim,
                    moe_intermediate_size, shared_expert_intermediate_size,
                    num_experts, num_experts_per_tok, rms_norm_eps,
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


class Qwen3_5_397B(Qwen3_5_Model):
    """Qwen3.5 397B-A17B with default config."""

    def __init__(self):
        super().__init__()
