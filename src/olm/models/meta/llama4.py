import torch
import torch.nn as nn

from olm.nn.structure import Block
from olm.nn.attention import GroupedQueryAttention, SlidingWindowAttention
from olm.nn.feedforward import SwiGLUFFN
from olm.nn.norms import RMSNorm
from olm.nn.embeddings import Embedding
from olm.nn.blocks import OutputHead
from olm.nn.moe import MoEFeedForward


class Llama4Block(Block):
    """
    A single decoder block for Llama 4 Maverick.

    Two independent per-layer choices, matching Llama 4's iRoPE design:

    - Attention: ``use_chunked_attention`` picks chunked local attention
      (approximated here with ``SlidingWindowAttention``) with RoPE; the
      alternative is a full/global layer with **no positional embedding**
      (NoPE, ``GroupedQueryAttention(use_rope=False)``) for long-context
      generalization.
    - Feed-forward: ``use_moe`` picks a sparse MoE layer (top-1 routed
      expert plus one always-active shared expert) over a dense SwiGLU
      layer.

        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length (for the RoPE cache).
        chunk_size (int): Local attention chunk size (approximated as a
            sliding window) for chunked-attention layers.
        intermediate_size (int): FFN hidden dim for dense layers and the
            shared expert.
        moe_intermediate_size (int): FFN hidden dim of each routed expert.
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts routed to per token.
        use_chunked_attention (bool): If True, local chunked attention with
            RoPE; else a global NoPE layer.
        use_moe (bool): Whether this layer is an MoE layer (else dense
            SwiGLU).
        dropout (float): Dropout probability.
        rope_theta (float): RoPE base frequency (chunked-attention layers
            only).
        rms_norm_eps (float): Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        chunk_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        use_chunked_attention: bool,
        use_moe: bool,
        dropout: float,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__([])
        self.use_moe = use_moe
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        if use_chunked_attention:
            self.self_attn = SlidingWindowAttention(
                embed_dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                window_size=chunk_size,
                head_dim=head_dim,
                dropout=dropout,
                rope_theta=rope_theta,
                use_bias=False,
            )
        else:
            self.self_attn = GroupedQueryAttention(
                embed_dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                head_dim=head_dim,
                dropout=dropout,
                use_bias=False,
                use_rope=False,
            )
        self.ffn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.last_router_logits = None

        if use_moe:
            self.ffn = MoEFeedForward(
                embed_dim=embed_dim,
                expert_cls=SwiGLUFFN,
                num_experts=num_experts,
                num_shared_experts=1,
                top_k=top_k,
                expert_kwargs={"hidden_dim": moe_intermediate_size, "bias": False},
                scoring_func="sigmoid",
                routing_method="topk",
                norm_weights=False,
            )
        else:
            self.ffn = SwiGLUFFN(embed_dim, hidden_dim=intermediate_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = x
        x = self.self_attn(
            self.attn_norm(x), **{k: v for k, v in kwargs.items() if k == "mask"}
        )
        x = residual + x

        residual = x
        if self.use_moe:
            x, router_logits = self.ffn(self.ffn_norm(x))
            self.last_router_logits = router_logits
        else:
            x = self.ffn(self.ffn_norm(x))
        return residual + x


class Llama4MaverickModel(Block):
    """
    Base class for Llama 4 Maverick.

    Structure:
        Embedding -> [Llama4Block] x N -> RMSNorm -> OutputHead.

    Two independent alternating patterns, both driven by layer index:

    - MoE/dense: every ``interleave_moe_layer_step``-th layer is a sparse
      MoE layer (top-1 routed + 1 shared expert); the rest are dense SwiGLU.
    - Attention: 3 out of every 4 layers use chunked local attention with
      RoPE; every 4th layer is a global, positionless (NoPE) layer.

    Llama 4 Maverick does not tie input/output embeddings; the named
    ``Llama4Maverick_400B`` preset passes ``tie_weights=False``.

    Notes:
        - This models the language-model component only; the reference
          checkpoint also includes an early-fusion vision encoder, omitted
          here.
        - Attention temperature tuning (used alongside NoPE for
          length generalization in the reference implementation) is not
          modeled; the NoPE/chunked alternation is applied statically.

    Forward:
        Accepts token IDs shaped ``[batch, seq_len]`` and returns logits
        shaped ``[batch, seq_len, vocab_size]``.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Model dimension.
        intermediate_size (int): FFN hidden dim for dense layers and shared
            experts.
        moe_intermediate_size (int): FFN hidden dim of each routed expert.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum context length.
        chunk_size (int): Local attention chunk size for chunked-attention
            layers.
        num_experts (int): Total number of routable experts.
        top_k (int): Number of experts routed to per token.
        interleave_moe_layer_step (int): Period of the dense:MoE pattern;
            every ``interleave_moe_layer_step``-th layer is MoE.
        nope_layer_interval (int): Period of the chunked:global(NoPE)
            attention pattern; every ``nope_layer_interval``-th layer is
            global/NoPE.
        rope_theta (float): RoPE base frequency (chunked-attention layers).
        dropout (float): Dropout probability.
        rms_norm_eps (float): Epsilon for RMSNorm layers.
        tie_weights (bool): Whether to tie the output head to the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        chunk_size: int,
        num_experts: int,
        top_k: int,
        interleave_moe_layer_step: int = 2,
        nope_layer_interval: int = 4,
        rope_theta: float = 500000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        embedding = Embedding(vocab_size, embed_dim)

        layers = nn.ModuleList(
            [
                Llama4Block(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    max_seq_len,
                    chunk_size,
                    intermediate_size,
                    moe_intermediate_size,
                    num_experts,
                    top_k,
                    use_chunked_attention=((layer_idx + 1) % nope_layer_interval != 0),
                    use_moe=(
                        layer_idx % interleave_moe_layer_step
                        == interleave_moe_layer_step - 1
                    ),
                    dropout=dropout,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        final_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        output_head = OutputHead(
            embed_dim,
            vocab_size,
            tied_embedding=embedding,
            tie_weights=tie_weights,
            use_norm=False,
        )

        super().__init__([embedding, final_norm, output_head])
        self.transformer_blocks = layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.blocks[0](x)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        x = self.blocks[1](x)
        return self.blocks[2](x)


class Llama4Maverick_400B(Llama4MaverickModel):
    """Llama 4 Maverick (400B total, 17B active; language-model component)."""

    def __init__(self):
        super().__init__(
            vocab_size=202048,
            embed_dim=5120,
            intermediate_size=8192,
            moe_intermediate_size=8192,
            num_layers=48,
            num_heads=40,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=1048576,
            chunk_size=8192,
            num_experts=128,
            top_k=1,
            interleave_moe_layer_step=2,
            nope_layer_interval=4,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            tie_weights=False,
        )
