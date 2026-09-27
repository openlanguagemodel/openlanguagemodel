"""Llama 2 and Llama 3 vs ``transformers.LlamaForCausalLM``.

Llama 2 is registered twice so that both OLM attention paths are exercised:
``Llama2Block`` picks ``FlashAttentionwithRoPE`` when ``num_kv_heads ==
num_heads`` and ``GroupedQueryAttention`` otherwise.

The output head is deliberately **untied** in the main cases. Released Llama 2
and Llama 3 checkpoints set ``tie_word_embeddings=False``, so an independent
``lm_head`` is the configuration that actually ships. Tying it would collapse
the embedding and the output projection into one parameter, and
``llama_family_map`` then has no ``lm_head.weight`` entry to carry -- meaning
the coverage check, the logit comparison and the gradient comparison would all
skip the output-head path entirely, and a conversion bug there could still
report clean parity. One tied case is kept alongside them so that path stays
covered too.
"""

from __future__ import annotations

from olm.models.meta.llama2 import Llama2Model
from olm.models.meta.llama3 import Llama3Model
from transformers import LlamaConfig, LlamaForCausalLM

from .._spec import ParityCase
from ._llama_family import llama_family_map

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
MAX_SEQ = 32
HEAD_DIM = EMBED // HEADS


def _config(
    *,
    num_kv_heads: int,
    rope_theta: float,
    rms_eps: float,
    tie_embeddings: bool = False,
) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=MAX_SEQ,
        rope_theta=rope_theta,
        rms_norm_eps=rms_eps,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=tie_embeddings,
        attn_implementation="eager",
    )


def _map(num_kv_heads: int, *, tied_embeddings: bool = False):
    def build(olm, hf):
        return llama_family_map(
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            tied_embeddings=tied_embeddings,
        )

    return build


# OLM hardcodes rms_norm_eps=1e-5 for both Llama 2 and Llama 3 blocks, which
# matches the LlamaConfig default.
RMS_EPS = 1e-5

CASES = [
    ParityCase(
        name="llama2-mha",
        reference="LlamaForCausalLM (num_kv_heads == num_heads)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=HEADS, rope_theta=10000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=10000.0,
            tie_weights=False,
        ),
        build_map=_map(HEADS),
        notes=(
            "Exercises OLM's FlashAttentionwithRoPE path (multi-head), with an "
            "untied output head as in released Llama 2 checkpoints."
        ),
    ),
    ParityCase(
        name="llama2-gqa",
        reference="LlamaForCausalLM (grouped-query)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=2, rope_theta=10000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=2,
            max_seq_len=MAX_SEQ,
            rope_theta=10000.0,
            tie_weights=False,
        ),
        build_map=_map(2),
        notes="Exercises OLM's GroupedQueryAttention path.",
    ),
    ParityCase(
        name="llama2-mha-tied",
        reference="LlamaForCausalLM (tie_word_embeddings=True)",
        build_hf=lambda: LlamaForCausalLM(
            _config(
                num_kv_heads=HEADS,
                rope_theta=10000.0,
                rms_eps=RMS_EPS,
                tie_embeddings=True,
            )
        ),
        build_olm=lambda: Llama2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=10000.0,
            tie_weights=True,
        ),
        build_map=_map(HEADS, tied_embeddings=True),
        notes=(
            "Covers OLM's tied-output-head path, where the embedding matrix is "
            "reused as the projection. Released Llama checkpoints are untied, so "
            "this is a configuration check rather than a checkpoint check."
        ),
        reference_deviations=(
            "tie_word_embeddings=True, whereas released Llama 2/3 checkpoints "
            "leave the output head untied.",
        ),
    ),
    ParityCase(
        name="llama3",
        reference="LlamaForCausalLM (rope_theta=500000)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=2, rope_theta=500000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama3Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=2,
            max_seq_len=MAX_SEQ,
            rope_theta=500000.0,
            tie_weights=False,
        ),
        build_map=_map(2),
        notes=(
            "Llama 3.1/3.2 checkpoints additionally use scaled RoPE for long "
            "context; OLM implements plain RoPE, so only the base rope_theta "
            "behaviour is compared here."
        ),
    ),
]
