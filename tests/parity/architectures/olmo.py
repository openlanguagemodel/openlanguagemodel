"""OLMo vs ``transformers.OlmoForCausalLM``.

Worth covering because it is the one architecture in the suite whose norms are
*non-parametric*: ``LayerNorm(elementwise_affine=False)`` on both sides, with no
learnable scale or shift anywhere. That means the whole model is exercised with
strictly fewer parameters than layers, which is a good check that the coverage
assertion is comparing modules rather than counting tensors.
"""

from __future__ import annotations

from olm.models.allenai.olmo import OLMoModel
from transformers import OlmoConfig, OlmoForCausalLM

from .._spec import MapEntry, ParityCase, cat_rows, rope_permute

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
MAX_SEQ = 32
HEAD_DIM = EMBED // HEADS
ROPE_THETA = 10000.0


def _config() -> OlmoConfig:
    return OlmoConfig(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        max_position_embeddings=MAX_SEQ,
        rope_theta=ROPE_THETA,
        hidden_act="silu",
        attention_bias=False,
        # OLMo 1 optionally clips QKV activations; OLM does not implement this,
        # and the released 1B/7B configs leave it unset.
        clip_qkv=None,
        pad_token_id=None,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def _build_map(olm, hf) -> list[MapEntry]:
    q_perm = rope_permute(HEADS, HEAD_DIM)

    entries = [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry("blocks.3.blocks.1.weight", "lm_head.weight"),
    ]

    for i in range(LAYERS):
        o = f"blocks.1.stack.{i}"
        h = f"model.layers.{i}"
        attn, mlp = f"{o}.blocks.0.block", f"{o}.blocks.1.block"

        entries += [
            MapEntry(f"{attn}.blocks.1.q_proj.weight", f"{h}.self_attn.q_proj.weight", q_perm),
            MapEntry(f"{attn}.blocks.1.k_proj.weight", f"{h}.self_attn.k_proj.weight", q_perm),
            MapEntry(f"{attn}.blocks.1.v_proj.weight", f"{h}.self_attn.v_proj.weight"),
            MapEntry(f"{attn}.blocks.1.out_proj.weight", f"{h}.self_attn.o_proj.weight"),
            MapEntry(
                f"{mlp}.blocks.1.up_proj.weight",
                (f"{h}.mlp.up_proj.weight", f"{h}.mlp.gate_proj.weight"),
                cat_rows,
            ),
            MapEntry(f"{mlp}.blocks.1.down_proj.weight", f"{h}.mlp.down_proj.weight"),
        ]

    return entries


CASES = [
    ParityCase(
        name="olmo",
        reference="OlmoForCausalLM",
        build_hf=lambda: OlmoForCausalLM(_config()),
        build_olm=lambda: OLMoModel(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            max_seq_len=MAX_SEQ,
            tie_weights=False,
        ),
        build_map=_build_map,
        notes="Non-parametric LayerNorm throughout; untied output head.",
    ),
]
