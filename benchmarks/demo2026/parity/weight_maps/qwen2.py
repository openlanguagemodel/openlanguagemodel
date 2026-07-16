"""HF Qwen2ForCausalLM -> OLM Qwen2Model weight map.

Same skeleton as Llama, plus Q/K/V projection biases (Qwen 2/2.5 uses
``attention_bias`` on Q/K/V only). Q/K weights *and biases* need the RoPE
row permutation from half-split to interleaved layout.
"""

from __future__ import annotations

import torch

from benchmarks.demo2026.parity.weight_maps._helpers import (
    MapEntry,
    WeightMap,
    permute_rope_rows,
)


def _identity(key: str):
    return lambda t: t[key]


def _permuted(key: str, num_heads: int):
    return lambda t: permute_rope_rows(t[key], num_heads)


def _swiglu_concat(up_key: str, gate_key: str):
    return lambda t: torch.cat([t[up_key], t[gate_key]], dim=0)


def build_map(config: dict) -> WeightMap:
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]

    entries = [
        MapEntry(
            "blocks.0.embedding.weight",
            ["model.embed_tokens.weight"],
            _identity("model.embed_tokens.weight"),
        ),
        MapEntry(
            "blocks.2.weight",
            ["model.norm.weight"],
            _identity("model.norm.weight"),
        ),
    ]

    for layer in range(num_layers):
        hf = f"model.layers.{layer}"
        olm = f"blocks.1.stack.{layer}"
        attn = f"{olm}.blocks.0.block"
        ffn = f"{olm}.blocks.1.block"
        entries.extend(
            [
                MapEntry(f"{attn}.blocks.0.weight", [f"{hf}.input_layernorm.weight"], _identity(f"{hf}.input_layernorm.weight")),
                MapEntry(f"{attn}.blocks.1.q_proj.weight", [f"{hf}.self_attn.q_proj.weight"], _permuted(f"{hf}.self_attn.q_proj.weight", num_heads)),
                MapEntry(f"{attn}.blocks.1.q_proj.bias", [f"{hf}.self_attn.q_proj.bias"], _permuted(f"{hf}.self_attn.q_proj.bias", num_heads)),
                MapEntry(f"{attn}.blocks.1.k_proj.weight", [f"{hf}.self_attn.k_proj.weight"], _permuted(f"{hf}.self_attn.k_proj.weight", num_kv_heads)),
                MapEntry(f"{attn}.blocks.1.k_proj.bias", [f"{hf}.self_attn.k_proj.bias"], _permuted(f"{hf}.self_attn.k_proj.bias", num_kv_heads)),
                MapEntry(f"{attn}.blocks.1.v_proj.weight", [f"{hf}.self_attn.v_proj.weight"], _identity(f"{hf}.self_attn.v_proj.weight")),
                MapEntry(f"{attn}.blocks.1.v_proj.bias", [f"{hf}.self_attn.v_proj.bias"], _identity(f"{hf}.self_attn.v_proj.bias")),
                MapEntry(f"{attn}.blocks.1.out_proj.weight", [f"{hf}.self_attn.o_proj.weight"], _identity(f"{hf}.self_attn.o_proj.weight")),
                MapEntry(f"{ffn}.blocks.0.weight", [f"{hf}.post_attention_layernorm.weight"], _identity(f"{hf}.post_attention_layernorm.weight")),
                MapEntry(
                    f"{ffn}.blocks.1.up_proj.weight",
                    [f"{hf}.mlp.up_proj.weight", f"{hf}.mlp.gate_proj.weight"],
                    _swiglu_concat(f"{hf}.mlp.up_proj.weight", f"{hf}.mlp.gate_proj.weight"),
                ),
                MapEntry(f"{ffn}.blocks.1.down_proj.weight", [f"{hf}.mlp.down_proj.weight"], _identity(f"{hf}.mlp.down_proj.weight")),
            ]
        )

    return WeightMap(entries)


HF_IGNORED = ["lm_head.weight"]
