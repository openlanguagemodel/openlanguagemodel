"""Gemma 2 vs ``transformers.Gemma2ForCausalLM``.

Gemma 2 is the most demanding case in the suite: sandwich normalisation,
``(1 + w)`` norm scales, embedding scaling by ``sqrt(hidden_size)``, attention
logit soft-capping, final logit soft-capping, a non-default query scale, and
sliding-window attention on alternating layers. ``seq_len`` is deliberately
larger than ``sliding_window`` so the window is actually exercised.

As with GPT-2, two cases are registered because OLM's GeGLU uses exact erf GELU
while Gemma 2's reference activation is the tanh approximation.
"""

from __future__ import annotations

from olm.models.google.gemma2 import Gemma2Model
from transformers import Gemma2Config, Gemma2ForCausalLM

from .._spec import MapEntry, ParityCase, add_scalar, cat_rows, identity, rope_permute

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
MAX_SEQ = 32
ROPE_THETA = 10000.0
SLIDING_WINDOW = 8
ATTN_SOFTCAP = 50.0
FINAL_SOFTCAP = 30.0
# int, because transformers 5.x validates this field as `int` and 16.0 raises
# StrictDataclassFieldValidationError; and != HEAD_DIM so that the query scale is
# actually exercised -- at 16 == HEAD_DIM, `query_pre_attn_scalar**-0.5` equals
# the default 1/sqrt(head_dim) and dropping it entirely is bit-identical.
QUERY_PRE_ATTN_SCALAR = 24


def _config(activation: str) -> Gemma2Config:
    return Gemma2Config(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_SEQ,
        rope_theta=ROPE_THETA,
        rms_norm_eps=1e-6,
        hidden_activation=activation,
        sliding_window=SLIDING_WINDOW,
        attn_logit_softcapping=ATTN_SOFTCAP,
        final_logit_softcapping=FINAL_SOFTCAP,
        query_pre_attn_scalar=QUERY_PRE_ATTN_SCALAR,
        attention_dropout=0.0,
        # No padding_idx: nn.Embedding(padding_idx=...) force-zeroes that row's
        # gradient in the reference, which OLM has no equivalent of. Leaving it
        # set would compare a zeroed row against a real one.
        pad_token_id=None,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def build_olm(**overrides) -> Gemma2Model:
    """Build the OLM model, optionally with one feature knob changed.

    The overrides exist for the negative-control tests, which break exactly one
    advertised feature at a time and require parity to collapse. Taking them
    here rather than restating the config in the test keeps the mutated model
    identical to the real one in every other respect -- otherwise a "the suite
    catches this" claim could rest on an unrelated difference.
    """
    kwargs = dict(
        vocab_size=VOCAB,
        embed_dim=EMBED,
        intermediate_size=INTERMEDIATE,
        num_layers=LAYERS,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ,
        rope_theta=ROPE_THETA,
        sliding_window=SLIDING_WINDOW,
        attn_logit_softcap=ATTN_SOFTCAP,
        final_logit_softcap=FINAL_SOFTCAP,
        query_pre_attn_scalar=QUERY_PRE_ATTN_SCALAR,
    )
    unknown = set(overrides) - set(kwargs)
    if unknown:
        raise TypeError(f"not Gemma 2 parity config keys: {sorted(unknown)}")
    kwargs.update(overrides)
    return Gemma2Model(**kwargs)


#: HF's Gemma2RMSNorm computes ``x_normed * (1 + weight)``; OLM's RMSNorm
#: computes ``x_normed * weight``. Folding the constant into the copied weight
#: makes the two identical without touching either implementation.
_gemma_norm = add_scalar(1.0)


def _build_map(olm, hf) -> list[MapEntry]:
    q_perm = rope_permute(HEADS, HEAD_DIM)
    k_perm = rope_permute(KV_HEADS, HEAD_DIM)

    entries = [
        MapEntry("blocks.0.embedding.weight", "model.embed_tokens.weight"),
        MapEntry("blocks.2.weight", "model.norm.weight", _gemma_norm, grad_fn=identity),
    ]

    for i in range(LAYERS):
        # Gemma2Block is a named module, so these paths read normally.
        o = f"blocks.1.stack.{i}"
        h = f"model.layers.{i}"

        entries += [
            MapEntry(
                f"{o}.input_layernorm.weight",
                f"{h}.input_layernorm.weight",
                _gemma_norm,
                grad_fn=identity,
            ),
            MapEntry(
                f"{o}.post_attention_layernorm.weight",
                f"{h}.post_attention_layernorm.weight",
                _gemma_norm,
                grad_fn=identity,
            ),
            MapEntry(
                f"{o}.pre_feedforward_layernorm.weight",
                f"{h}.pre_feedforward_layernorm.weight",
                _gemma_norm,
                grad_fn=identity,
            ),
            MapEntry(
                f"{o}.post_feedforward_layernorm.weight",
                f"{h}.post_feedforward_layernorm.weight",
                _gemma_norm,
                grad_fn=identity,
            ),
            MapEntry(f"{o}.self_attn.q_proj.weight", f"{h}.self_attn.q_proj.weight", q_perm),
            MapEntry(f"{o}.self_attn.k_proj.weight", f"{h}.self_attn.k_proj.weight", k_perm),
            MapEntry(f"{o}.self_attn.v_proj.weight", f"{h}.self_attn.v_proj.weight"),
            MapEntry(f"{o}.self_attn.out_proj.weight", f"{h}.self_attn.o_proj.weight"),
            MapEntry(
                f"{o}.mlp.up_proj.weight",
                (f"{h}.mlp.up_proj.weight", f"{h}.mlp.gate_proj.weight"),
                cat_rows,
                note="OLM GeGLU computes value * gelu(gate); value half comes first.",
            ),
            MapEntry(f"{o}.mlp.down_proj.weight", f"{h}.mlp.down_proj.weight"),
        ]

    return entries


CASES = [
    ParityCase(
        name="gemma2",
        reference="Gemma2ForCausalLM",
        build_hf=lambda: Gemma2ForCausalLM(_config("gelu")),
        build_olm=build_olm,
        build_map=_build_map,
        notes=(
            "Sandwich norms, (1+w) norm scales, sqrt(hidden) embedding scale, "
            "attention and final logit soft-capping, alternating sliding window."
        ),
        reference_deviations=(
            "hidden_activation='gelu' (exact erf) instead of the Gemma 2 default "
            "'gelu_pytorch_tanh'; OLM's GeGLU calls F.gelu with no approximation",
        ),
    ),
    ParityCase(
        name="gemma2-stock",
        reference="Gemma2ForCausalLM (stock config)",
        build_hf=lambda: Gemma2ForCausalLM(_config("gelu_pytorch_tanh")),
        build_olm=build_olm,
        build_map=_build_map,
        notes="Reference at its real default activation. Expected to disagree.",
    ),
]
