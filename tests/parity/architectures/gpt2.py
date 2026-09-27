"""GPT-2 vs ``transformers.GPT2LMHeadModel``.

Two cases are registered on purpose:

``gpt2``
    Reference forced to exact (erf) GELU so that everything *except* the
    activation is measured. The deviation is recorded, not hidden.

``gpt2-stock``
    Reference left at its real default, ``gelu_new`` (the tanh approximation
    that the released GPT-2 checkpoints were trained with). This is the honest
    answer to "does OLM reproduce GPT-2", and it does not match.
"""

from __future__ import annotations

from olm.models.openai.gpt2 import GPT2Model
from transformers import GPT2Config, GPT2LMHeadModel

from .._spec import MapEntry, ParityCase, compose, rows, transpose

VOCAB = 61
EMBED = 64
LAYERS = 2
HEADS = 4
MAX_SEQ = 32


def _config(activation: str) -> GPT2Config:
    return GPT2Config(
        vocab_size=VOCAB,
        n_embd=EMBED,
        n_layer=LAYERS,
        n_head=HEADS,
        n_positions=MAX_SEQ,
        n_inner=4 * EMBED,
        activation_function=activation,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _build_olm() -> GPT2Model:
    return GPT2Model(
        vocab_size=VOCAB,
        embed_dim=EMBED,
        num_layers=LAYERS,
        num_heads=HEADS,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
        tie_weights=True,
    )


def _build_map(olm, hf) -> list[MapEntry]:
    d = EMBED
    entries = [
        MapEntry("blocks.0.blocks.0.embedding.weight", "transformer.wte.weight"),
        MapEntry("blocks.0.blocks.1.pos_embedding.weight", "transformer.wpe.weight"),
        MapEntry("blocks.2.blocks.0.gamma", "transformer.ln_f.weight"),
        MapEntry("blocks.2.blocks.0.beta", "transformer.ln_f.bias"),
    ]

    for i in range(LAYERS):
        o = f"blocks.1.stack.{i}"
        h = f"transformer.h.{i}"
        attn, mlp = f"{o}.blocks.0.block", f"{o}.blocks.1.block"

        # HF fuses Q/K/V into one Conv1D storing [in_features, 3 * out_features];
        # OLM keeps three nn.Linear layers storing [out_features, in_features].
        qkv_w = lambda lo, hi: compose(rows(lo, hi), transpose)
        qkv_b = lambda lo, hi: rows(lo, hi)

        entries += [
            MapEntry(f"{attn}.blocks.0.gamma", f"{h}.ln_1.weight"),
            MapEntry(f"{attn}.blocks.0.beta", f"{h}.ln_1.bias"),
            MapEntry(f"{attn}.blocks.1.q_proj.weight", f"{h}.attn.c_attn.weight", qkv_w(0, d)),
            MapEntry(f"{attn}.blocks.1.q_proj.bias", f"{h}.attn.c_attn.bias", qkv_b(0, d)),
            MapEntry(f"{attn}.blocks.1.k_proj.weight", f"{h}.attn.c_attn.weight", qkv_w(d, 2 * d)),
            MapEntry(f"{attn}.blocks.1.k_proj.bias", f"{h}.attn.c_attn.bias", qkv_b(d, 2 * d)),
            MapEntry(f"{attn}.blocks.1.v_proj.weight", f"{h}.attn.c_attn.weight", qkv_w(2 * d, 3 * d)),
            MapEntry(f"{attn}.blocks.1.v_proj.bias", f"{h}.attn.c_attn.bias", qkv_b(2 * d, 3 * d)),
            MapEntry(f"{attn}.blocks.1.out_proj.weight", f"{h}.attn.c_proj.weight", transpose),
            MapEntry(f"{attn}.blocks.1.out_proj.bias", f"{h}.attn.c_proj.bias"),
            MapEntry(f"{mlp}.blocks.0.gamma", f"{h}.ln_2.weight"),
            MapEntry(f"{mlp}.blocks.0.beta", f"{h}.ln_2.bias"),
            MapEntry(f"{mlp}.blocks.1.up_proj.weight", f"{h}.mlp.c_fc.weight", transpose),
            MapEntry(f"{mlp}.blocks.1.up_proj.bias", f"{h}.mlp.c_fc.bias"),
            MapEntry(f"{mlp}.blocks.1.down_proj.weight", f"{h}.mlp.c_proj.weight", transpose),
            MapEntry(f"{mlp}.blocks.1.down_proj.bias", f"{h}.mlp.c_proj.bias"),
        ]

    return entries


CASES = [
    ParityCase(
        name="gpt2",
        reference="GPT2LMHeadModel",
        build_hf=lambda: GPT2LMHeadModel(_config("gelu")),
        build_olm=_build_olm,
        build_map=_build_map,
        notes="Learned absolute positions, LayerNorm, fused Conv1D QKV.",
        reference_deviations=(
            "activation_function='gelu' (exact erf) instead of the GPT-2 default "
            "'gelu_new'; OLM's ClassicFFN hardcodes nn.GELU(approximate='none')",
        ),
    ),
    ParityCase(
        name="gpt2-stock",
        reference="GPT2LMHeadModel (stock config)",
        build_hf=lambda: GPT2LMHeadModel(_config("gelu_new")),
        build_olm=_build_olm,
        build_map=_build_map,
        notes="Reference at its real default activation. Expected to disagree.",
    ),
]
