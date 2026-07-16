"""HF GPT2LMHeadModel -> OLM GPT2Model weight map.

HF layout notes:
  - ``transformer.h.{L}.attn.c_attn`` is a fused Conv1D with weight shape
    ``(embed, 3*embed)`` holding Q, K, V in that order; it must be transposed
    and split.
  - All Conv1D weights (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) are stored
    as ``(in, out)`` and need a transpose for ``nn.Linear`` semantics.
  - ``lm_head.weight`` is tied to ``transformer.wte.weight``; it is an alias
    and is never copied separately.
"""

from __future__ import annotations

import torch

from benchmarks.demo2026.parity.weight_maps._helpers import (
    MapEntry,
    WeightMap,
    transpose_conv1d,
)


def _split_qkv_weight(index: int):
    def build(t: dict) -> torch.Tensor:
        (key,) = t.keys()
        weight = transpose_conv1d(t[key])  # (3*embed, embed)
        return weight.chunk(3, dim=0)[index].contiguous()

    return build


def _split_qkv_bias(index: int):
    def build(t: dict) -> torch.Tensor:
        (key,) = t.keys()
        return t[key].chunk(3, dim=0)[index].contiguous()

    return build


def _transposed(key: str):
    return lambda t: transpose_conv1d(t[key])


def _identity(key: str):
    return lambda t: t[key]


def build_map(config: dict) -> WeightMap:
    num_layers = config["num_layers"]
    entries = [
        MapEntry(
            "blocks.0.blocks.0.embedding.weight",
            ["transformer.wte.weight"],
            _identity("transformer.wte.weight"),
        ),
        MapEntry(
            "blocks.0.blocks.1.pos_embedding.weight",
            ["transformer.wpe.weight"],
            _identity("transformer.wpe.weight"),
        ),
        MapEntry(
            "blocks.2.blocks.0.gamma",
            ["transformer.ln_f.weight"],
            _identity("transformer.ln_f.weight"),
        ),
        MapEntry(
            "blocks.2.blocks.0.beta",
            ["transformer.ln_f.bias"],
            _identity("transformer.ln_f.bias"),
        ),
    ]

    for layer in range(num_layers):
        hf = f"transformer.h.{layer}"
        olm = f"blocks.1.stack.{layer}"
        attn = f"{olm}.blocks.0.block"
        ffn = f"{olm}.blocks.1.block"
        entries.extend(
            [
                MapEntry(f"{attn}.blocks.0.gamma", [f"{hf}.ln_1.weight"], _identity(f"{hf}.ln_1.weight")),
                MapEntry(f"{attn}.blocks.0.beta", [f"{hf}.ln_1.bias"], _identity(f"{hf}.ln_1.bias")),
                MapEntry(f"{attn}.blocks.1.q_proj.weight", [f"{hf}.attn.c_attn.weight"], _split_qkv_weight(0)),
                MapEntry(f"{attn}.blocks.1.k_proj.weight", [f"{hf}.attn.c_attn.weight"], _split_qkv_weight(1)),
                MapEntry(f"{attn}.blocks.1.v_proj.weight", [f"{hf}.attn.c_attn.weight"], _split_qkv_weight(2)),
                MapEntry(f"{attn}.blocks.1.q_proj.bias", [f"{hf}.attn.c_attn.bias"], _split_qkv_bias(0)),
                MapEntry(f"{attn}.blocks.1.k_proj.bias", [f"{hf}.attn.c_attn.bias"], _split_qkv_bias(1)),
                MapEntry(f"{attn}.blocks.1.v_proj.bias", [f"{hf}.attn.c_attn.bias"], _split_qkv_bias(2)),
                MapEntry(f"{attn}.blocks.1.out_proj.weight", [f"{hf}.attn.c_proj.weight"], _transposed(f"{hf}.attn.c_proj.weight")),
                MapEntry(f"{attn}.blocks.1.out_proj.bias", [f"{hf}.attn.c_proj.bias"], _identity(f"{hf}.attn.c_proj.bias")),
                MapEntry(f"{ffn}.blocks.0.gamma", [f"{hf}.ln_2.weight"], _identity(f"{hf}.ln_2.weight")),
                MapEntry(f"{ffn}.blocks.0.beta", [f"{hf}.ln_2.bias"], _identity(f"{hf}.ln_2.bias")),
                MapEntry(f"{ffn}.blocks.1.up_proj.weight", [f"{hf}.mlp.c_fc.weight"], _transposed(f"{hf}.mlp.c_fc.weight")),
                MapEntry(f"{ffn}.blocks.1.up_proj.bias", [f"{hf}.mlp.c_fc.bias"], _identity(f"{hf}.mlp.c_fc.bias")),
                MapEntry(f"{ffn}.blocks.1.down_proj.weight", [f"{hf}.mlp.c_proj.weight"], _transposed(f"{hf}.mlp.c_proj.weight")),
                MapEntry(f"{ffn}.blocks.1.down_proj.bias", [f"{hf}.mlp.c_proj.bias"], _identity(f"{hf}.mlp.c_proj.bias")),
            ]
        )

    return WeightMap(entries)


# lm_head.weight is an alias of transformer.wte.weight (tied); never copied.
HF_IGNORED = ["lm_head.weight"]
