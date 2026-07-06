import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import MultiHeadLatentAttention, LightningAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead
from olm.nn.moe import MoEFeedForward


class Ling2_5_MLA_DenseBlock(Block):
    """MLA block with a dense FFN (used in the first 4 layers)."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        max_seq_len: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(
            embed_dim, num_heads, max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x
        residual = x
        x = self.ffn(self.ffn_norm(x))
        return residual + x


class Ling2_5_MLA_MoEBlock(Block):
    """
    MLA block with MoE FFN.

    Multi-Head Latent Attention with KV compression (rank=512) paired
    with 256-expert sigmoid-routed MoE (noaux_tc, top-8).
    """

    def __init__(
        self,
        embed_dim: int,
        moe_intermediate_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        max_seq_len: int,
        num_experts: int,
        num_experts_per_tok: int,
        rope_theta: float,
        rms_norm_eps: float,
        routed_scaling_factor: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(
            embed_dim, num_heads, max_seq_len,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            rope_theta=rope_theta, rms_norm_eps=rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
            scoring_func="sigmoid",
            routing_method="noaux_tc",
            use_router_bias=True,
            norm_weights=True,
            fp32_gate=True,
            routed_scaling_factor=routed_scaling_factor,
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


class Ling2_5_Lightning_DenseBlock(Block):
    """Lightning Attention block with a dense FFN (used in first 4 layers)."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = LightningAttention(
            embed_dim, num_heads, head_dim=head_dim,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"})
        x = residual + x
        residual = x
        x = self.ffn(self.ffn_norm(x))
        return residual + x


class Ling2_5_Lightning_MoEBlock(Block):
    """
    Lightning Attention block with MoE FFN.

    O(N) linear attention with 256-expert sigmoid-routed MoE.
    """

    def __init__(
        self,
        embed_dim: int,
        moe_intermediate_size: int,
        num_heads: int,
        head_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
        routed_scaling_factor: float,
    ):
        super().__init__([])
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.self_attn = LightningAttention(
            embed_dim, num_heads, head_dim=head_dim,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.moe = MoEFeedForward(
            embed_dim=embed_dim,
            expert_cls=SwiGLUFFN,
            num_experts=num_experts,
            num_shared_experts=1,
            top_k=num_experts_per_tok,
            expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
            scoring_func="sigmoid",
            routing_method="noaux_tc",
            use_router_bias=True,
            norm_weights=True,
            fp32_gate=True,
            routed_scaling_factor=routed_scaling_factor,
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


class Ling2_5_Model(Block):
    """
    Ling 2.5 1T -- trillion-parameter hybrid with Lightning Attention + MLA.

    80 layers arranged in groups of 8: each group has 1 MLA layer + 7
    Lightning Attention layers (7:1 ratio).  The first 4 layers use
    dense SwiGLU FFN; all others use 256-expert sigmoid-routed MoE
    (noaux_tc, top-8, routed_scaling=2.5).

    MLA uses KV compression (rank=512) and Q compression (rank=1536)
    with 64 heads, split head dim (128 nope + 64 rope).  Lightning
    Attention uses O(N) linear attention with 64 heads.

    Structure:
        Embedding -> [MLA | Lightning] x 80 -> RMSNorm -> OutputHead

    Forward:
        Token IDs ``[batch, seq_len]`` -> logits ``[batch, seq_len, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 157184,
        embed_dim: int = 8192,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_layers: int = 80,
        num_heads: int = 64,
        head_dim: int = 128,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        max_seq_len: int = 131072,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        layer_group_size: int = 8,
        first_k_dense: int = 4,
        rope_theta: float = 6000000.0,
        rms_norm_eps: float = 1e-6,
        routed_scaling_factor: float = 2.5,
        tie_weights: bool = False,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layers = nn.ModuleList()
        for i in range(num_layers):
            is_mla_layer = (i % layer_group_size == 0)
            is_dense = (i < first_k_dense)

            if is_mla_layer:
                if is_dense:
                    layers.append(Ling2_5_MLA_DenseBlock(
                        embed_dim, intermediate_size, num_heads,
                        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                        v_head_dim, q_lora_rank,
                        max_seq_len, rope_theta, rms_norm_eps,
                    ))
                else:
                    layers.append(Ling2_5_MLA_MoEBlock(
                        embed_dim, moe_intermediate_size, num_heads,
                        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                        v_head_dim, q_lora_rank,
                        max_seq_len, num_experts, num_experts_per_tok,
                        rope_theta, rms_norm_eps, routed_scaling_factor,
                    ))
            else:
                if is_dense:
                    layers.append(Ling2_5_Lightning_DenseBlock(
                        embed_dim, intermediate_size, num_heads,
                        head_dim, rms_norm_eps,
                    ))
                else:
                    layers.append(Ling2_5_Lightning_MoEBlock(
                        embed_dim, moe_intermediate_size, num_heads,
                        head_dim, num_experts, num_experts_per_tok,
                        rms_norm_eps, routed_scaling_factor,
                    ))

        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim, vocab_size,
            tied_embedding=embedding if tie_weights else None,
            tie_weights=tie_weights, use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class Ling2_5_1T(Ling2_5_Model):
    """Ling 2.5 1T with default config (63B active of 1T total)."""

    def __init__(self):
        super().__init__()
