"""Qwen3 (flagship MoE) vs ``transformers.Qwen3MoeForCausalLM``.

The only sparse architecture in the suite, and the only one with QK-norm. Both
are worth pinning down: MoE routing is easy to get subtly wrong, and a routing
difference shows up as a large logit difference rather than a small one.

OLM's router and Qwen3's agree on ordering -- softmax over *all* experts, then
top-k, then renormalise over the selected k -- so no transform is needed on the
gate. Expert weights carry the same gated-MLP half-swap as the dense Llama
family, from either of the two reference layouts: one Linear per expert before
transformers 5, or two fused 3-D tensors per layer from 5 on.
"""

from __future__ import annotations

import torch
from olm.models.alibaba.qwen3 import Qwen3Model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from .._spec import MapEntry, ParityCase, cat_rows, rope_permute

VOCAB = 61
EMBED = 64
MOE_INTERMEDIATE = 24
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
MAX_SEQ = 32
NUM_EXPERTS = 4
TOP_K = 2
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6


def _config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=MOE_INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        # Every layer is sparse in Qwen3-235B-A22B, which is what OLM models.
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=MAX_SEQ,
        rope_theta=ROPE_THETA,
        rms_norm_eps=RMS_EPS,
        hidden_act="silu",
        attention_dropout=0.0,
        use_sliding_window=False,
        pad_token_id=None,
        tie_word_embeddings=False,
        attn_implementation="eager",
        # The default grouped-mm expert kernel rejects Double, so the float64
        # case dies with "Expected mat_a to be Float32, BFloat16 or Float16
        # matrix, got Double" without this. Same intent as attn_implementation.
        experts_implementation="eager",
    )


def _build_map(olm, hf) -> list[MapEntry]:
    q_perm = rope_permute(HEADS, HEAD_DIM)
    k_perm = rope_permute(KV_HEADS, HEAD_DIM)
    # A single head's worth of the same permutation, for the per-head QK-norm.
    head_perm = rope_permute(1, HEAD_DIM)

    # transformers >= 5 fuses the experts into two 3-D tensors per layer --
    # `mlp.experts.gate_up_proj` [E, 2I, H] in (gate, up) row order and
    # `mlp.experts.down_proj` [E, H, I] -- instead of one Linear per expert.
    # #77 adds a minimum-versions job at transformers 4.57.1, which still has
    # the unfused layout, so both have to be supported. Detected from the
    # reference's own parameter names rather than a version check.
    hf_params = dict(hf.named_parameters())
    fused_experts = "model.layers.0.mlp.experts.gate_up_proj" in hf_params

    entries = [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry("blocks.2.weight", "model.norm.weight"),
        MapEntry("blocks.3.blocks.1.weight", "lm_head.weight"),
    ]

    for i in range(LAYERS):
        o = f"blocks.1.stack.{i}"
        h = f"model.layers.{i}"
        attn, moe = f"{o}.blocks.0.block", f"{o}.blocks.1.block"

        entries += [
            MapEntry(f"{attn}.blocks.0.weight", f"{h}.input_layernorm.weight"),
            MapEntry(f"{attn}.blocks.1.q_proj.weight", f"{h}.self_attn.q_proj.weight", q_perm),
            MapEntry(f"{attn}.blocks.1.k_proj.weight", f"{h}.self_attn.k_proj.weight", k_perm),
            MapEntry(f"{attn}.blocks.1.v_proj.weight", f"{h}.self_attn.v_proj.weight"),
            MapEntry(f"{attn}.blocks.1.out_proj.weight", f"{h}.self_attn.o_proj.weight"),
            # QK-norm is applied per head *after* the Q/K projection and before
            # RoPE, so on the OLM side it sees the head axis already in the
            # interleaved layout. Its per-head scale must be permuted to match.
            # A forward-only check cannot catch this: transformers initialises
            # these scales to exactly ones, and a constant vector is invariant
            # under permutation.
            MapEntry(
                f"{attn}.blocks.1.q_norm.weight",
                f"{h}.self_attn.q_norm.weight",
                head_perm,
            ),
            MapEntry(
                f"{attn}.blocks.1.k_norm.weight",
                f"{h}.self_attn.k_norm.weight",
                head_perm,
            ),
            MapEntry(f"{moe}.blocks.0.weight", f"{h}.post_attention_layernorm.weight"),
            MapEntry(f"{moe}.blocks.1.router.gate.weight", f"{h}.mlp.gate.weight"),
        ]

        for e in range(NUM_EXPERTS):
            if fused_experts:
                entries += [
                    MapEntry(
                        f"{moe}.blocks.1.experts.{e}.up_proj.weight",
                        f"{h}.mlp.experts.gate_up_proj",
                        # Slice out expert e, then the same half-swap as the
                        # unfused path: the reference stores (gate, up) and
                        # OLM's fused tensor is (up, gate). Linear, so it is
                        # also the correct gradient transform.
                        lambda t, e=e: torch.cat(
                            [t[e, MOE_INTERMEDIATE:], t[e, :MOE_INTERMEDIATE]]
                        ).contiguous(),
                        note="fused [E, 2I, H] experts (transformers >= 5)",
                    ),
                    MapEntry(
                        f"{moe}.blocks.1.experts.{e}.down_proj.weight",
                        f"{h}.mlp.experts.down_proj",
                        lambda t, e=e: t[e].contiguous(),
                        note="fused [E, H, I] experts (transformers >= 5)",
                    ),
                ]
            else:
                entries += [
                    MapEntry(
                        f"{moe}.blocks.1.experts.{e}.up_proj.weight",
                        (
                            f"{h}.mlp.experts.{e}.up_proj.weight",
                            f"{h}.mlp.experts.{e}.gate_proj.weight",
                        ),
                        cat_rows,
                    ),
                    MapEntry(
                        f"{moe}.blocks.1.experts.{e}.down_proj.weight",
                        f"{h}.mlp.experts.{e}.down_proj.weight",
                    ),
                ]

    return entries


CASES = [
    ParityCase(
        name="qwen3-moe",
        reference="Qwen3MoeForCausalLM",
        build_hf=lambda: Qwen3MoeForCausalLM(_config()),
        build_olm=lambda: Qwen3Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            moe_intermediate_size=MOE_INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            max_seq_len=MAX_SEQ,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            head_dim=HEAD_DIM,
            rope_theta=ROPE_THETA,
            rms_norm_eps=RMS_EPS,
            tie_weights=False,
        ),
        build_map=_build_map,
        notes=(
            "Sparse top-k MoE routing plus per-head QK-norm. Compared on the "
            "language-model loss only -- see reference_deviations for why the "
            "router auxiliary objective is out of scope here."
        ),
        reference_deviations=(
            "The router load-balancing auxiliary loss is NOT compared. Both "
            "sides are scored with plain next-token cross entropy computed from "
            "logits, which excludes the auxiliary objective and its router "
            "gradients. Qwen3MoeForCausalLM can return router logits and add its "
            "configured aux loss, but OLM's Qwen3Model routes through "
            "SwiGLUMoEFFN -> MoEFeedForwardBase, whose forward() returns only the "
            "output tensor and discards the router logits, so there is nothing to "
            "compare against. Read this case as forward and LM-loss parity, not "
            "as evidence that real MoE training is equivalent: with no aux loss "
            "the two can agree here and still diverge in practice, since load "
            "balancing is what stops the router collapsing onto a few experts. "
            "OLM's newer olm.nn.moe.MoEFFN does return router logits.",
        ),
    ),
]
