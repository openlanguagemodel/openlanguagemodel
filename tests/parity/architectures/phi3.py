"""Phi-3 vs ``transformers.Phi3ForCausalLM``.

Phi-3 is the mirror image of GPT-2's fusion problem: here it is the *reference*
that fuses, keeping a single ``qkv_proj`` and a single ``gate_up_proj``, while
OLM keeps them split (except for its own gated-MLP fusion, which uses the
opposite half-ordering). Both fusions are unpicked explicitly below.
"""

from __future__ import annotations

from olm.models.microsoft.phi3 import Phi3Model
from transformers import Phi3Config, Phi3ForCausalLM

from .._spec import MapEntry, ParityCase, cat_rows, compose, rope_permute, rows

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
KV_HEADS = 2
MAX_SEQ = 32
HEAD_DIM = EMBED // HEADS
ROPE_THETA = 10000.0
RMS_EPS = 1e-5  # OLM hardcodes 1e-5 in Phi3Block; matches the Phi3Config default.


def _config() -> Phi3Config:
    return Phi3Config(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=MAX_SEQ,
        original_max_position_embeddings=MAX_SEQ,
        rope_theta=ROPE_THETA,
        rope_scaling=None,
        rms_norm_eps=RMS_EPS,
        hidden_act="silu",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        # See gemma2.py: padding_idx would zero one embedding row's gradient
        # in the reference only.
        pad_token_id=None,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _build_map(olm, hf) -> list[MapEntry]:
    q_end = HEADS * HEAD_DIM
    k_end = q_end + KV_HEADS * HEAD_DIM
    v_end = k_end + KV_HEADS * HEAD_DIM

    q_perm = rope_permute(HEADS, HEAD_DIM)
    k_perm = rope_permute(KV_HEADS, HEAD_DIM)

    entries = [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry("blocks.2.weight", "model.norm.weight"),
    ]

    for i in range(LAYERS):
        o = f"blocks.1.stack.{i}"
        h = f"model.layers.{i}"
        attn, mlp = f"{o}.blocks.0.block", f"{o}.blocks.1.block"

        entries += [
            MapEntry(f"{attn}.blocks.0.weight", f"{h}.input_layernorm.weight"),
            MapEntry(
                f"{attn}.blocks.1.q_proj.weight",
                f"{h}.self_attn.qkv_proj.weight",
                compose(q_perm, rows(0, q_end)),
                note="unfuse HF qkv_proj, then apply the RoPE head permutation",
            ),
            MapEntry(
                f"{attn}.blocks.1.k_proj.weight",
                f"{h}.self_attn.qkv_proj.weight",
                compose(k_perm, rows(q_end, k_end)),
            ),
            MapEntry(
                f"{attn}.blocks.1.v_proj.weight",
                f"{h}.self_attn.qkv_proj.weight",
                rows(k_end, v_end),
            ),
            MapEntry(f"{attn}.blocks.1.out_proj.weight", f"{h}.self_attn.o_proj.weight"),
            MapEntry(f"{mlp}.blocks.0.weight", f"{h}.post_attention_layernorm.weight"),
            MapEntry(
                f"{mlp}.blocks.1.up_proj.weight",
                f"{h}.mlp.gate_up_proj.weight",
                lambda t: cat_rows(
                    rows(INTERMEDIATE, 2 * INTERMEDIATE)(t), rows(0, INTERMEDIATE)(t)
                ),
                note=(
                    "HF gate_up_proj chunks as (gate, up) and computes up * silu(gate); "
                    "OLM's SwiGLU chunks as (value, gate) and computes value * silu(gate). "
                    "The two halves are therefore swapped."
                ),
            ),
            MapEntry(f"{mlp}.blocks.1.down_proj.weight", f"{h}.mlp.down_proj.weight"),
        ]

    return entries


CASES = [
    ParityCase(
        name="phi3",
        reference="Phi3ForCausalLM",
        build_hf=lambda: Phi3ForCausalLM(_config()),
        build_olm=lambda: Phi3Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=ROPE_THETA,
            activation="swiglu",
        ),
        build_map=_build_map,
        notes=(
            "Released Phi-3 checkpoints use LongRoPE scaling for long context; "
            "OLM implements plain RoPE, so rope_scaling is disabled on the "
            "reference and only base RoPE behaviour is compared."
        ),
    ),
]
