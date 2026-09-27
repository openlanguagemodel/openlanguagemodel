"""Shared weight map for the pre-norm / RoPE / gated-MLP family.

Llama 2, Llama 3, Qwen 2.5 and Mistral all compile to the same OLM block shape:

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

so they share one mapping, parameterised by head counts and whether the Q/K/V
projections carry biases. The OLM parameter paths look positional because OLM
builds models out of anonymous ``Block``/``Repeat``/``Residual`` combinators
rather than named modules; the indices are spelled out here once.

OLM layer ``i`` path layout::

    blocks.1.stack.{i}.blocks.0.block.blocks.0   input RMSNorm
    blocks.1.stack.{i}.blocks.0.block.blocks.1   attention
    blocks.1.stack.{i}.blocks.1.block.blocks.0   post-attention RMSNorm
    blocks.1.stack.{i}.blocks.1.block.blocks.1   SwiGLU MLP
"""

from __future__ import annotations

from .._spec import MapEntry, cat_rows, rope_permute


def llama_family_map(
    *,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qkv_bias: bool = False,
    tied_embeddings: bool = True,
) -> list[MapEntry]:
    q_perm = rope_permute(num_heads, head_dim)
    k_perm = rope_permute(num_kv_heads, head_dim)

    entries: list[MapEntry] = [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry("blocks.2.weight", "model.norm.weight"),
    ]

    if not tied_embeddings:
        entries.append(MapEntry("blocks.3.blocks.1.weight", "lm_head.weight"))

    for i in range(num_layers):
        olm = f"blocks.1.stack.{i}"
        hf = f"model.layers.{i}"
        attn, mlp = f"{olm}.blocks.0.block", f"{olm}.blocks.1.block"

        entries += [
            MapEntry(f"{attn}.blocks.0.weight", f"{hf}.input_layernorm.weight"),
            MapEntry(
                f"{attn}.blocks.1.q_proj.weight",
                f"{hf}.self_attn.q_proj.weight",
                q_perm,
                note="head-axis permutation: HF rotate_half -> OLM interleaved RoPE",
            ),
            MapEntry(
                f"{attn}.blocks.1.k_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                k_perm,
                note="head-axis permutation: HF rotate_half -> OLM interleaved RoPE",
            ),
            MapEntry(
                f"{attn}.blocks.1.v_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
                note="values are not rotated, so no permutation",
            ),
            MapEntry(f"{attn}.blocks.1.out_proj.weight", f"{hf}.self_attn.o_proj.weight"),
            MapEntry(f"{mlp}.blocks.0.weight", f"{hf}.post_attention_layernorm.weight"),
            MapEntry(
                f"{mlp}.blocks.1.up_proj.weight",
                (f"{hf}.mlp.up_proj.weight", f"{hf}.mlp.gate_proj.weight"),
                cat_rows,
                note=(
                    "OLM fuses the gated MLP into one projection and splits it as "
                    "(value, gate) with value * silu(gate); HF keeps up_proj and "
                    "gate_proj separate as silu(gate) * up. Value half comes first."
                ),
            ),
            MapEntry(f"{mlp}.blocks.1.down_proj.weight", f"{hf}.mlp.down_proj.weight"),
        ]

        if qkv_bias:
            entries += [
                MapEntry(
                    f"{attn}.blocks.1.q_proj.bias", f"{hf}.self_attn.q_proj.bias", q_perm
                ),
                MapEntry(
                    f"{attn}.blocks.1.k_proj.bias", f"{hf}.self_attn.k_proj.bias", k_perm
                ),
                MapEntry(f"{attn}.blocks.1.v_proj.bias", f"{hf}.self_attn.v_proj.bias"),
            ]

    return entries
