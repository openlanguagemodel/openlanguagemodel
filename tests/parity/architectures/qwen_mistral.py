"""Qwen 2.5 and Mistral vs their ``transformers`` references.

Both reuse the Llama-family block shape. Qwen 2.5 differs only in carrying
biases on Q/K/V (but not on the output projection); Mistral Small 3.1 differs
only in leaving the input and output embeddings untied.
"""

from __future__ import annotations

from olm.models.alibaba.qwen2 import Qwen2Model
from olm.models.mistralai.mistral_small3_1 import MistralSmall3_1_Model
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from .._spec import ParityCase
from ._llama_family import llama_family_map

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
KV_HEADS = 2
MAX_SEQ = 32
HEAD_DIM = EMBED // HEADS

QWEN_ROPE_THETA = 1000000.0
QWEN_RMS_EPS = 1e-6
MISTRAL_ROPE_THETA = 1000000.0
MISTRAL_RMS_EPS = 1e-5


def _qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=MAX_SEQ,
        rope_theta=QWEN_ROPE_THETA,
        rms_norm_eps=QWEN_RMS_EPS,
        hidden_act="silu",
        # Qwen 2.5 attends over the full context at this scale; the config's
        # sliding-window machinery is off by default in transformers 4.44.
        use_sliding_window=False,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _mistral_config() -> MistralConfig:
    return MistralConfig(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=MAX_SEQ,
        rope_theta=MISTRAL_ROPE_THETA,
        rms_norm_eps=MISTRAL_RMS_EPS,
        hidden_act="silu",
        # Mistral Small 3.1 dropped the sliding window that Mistral 7B used.
        sliding_window=None,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


CASES = [
    ParityCase(
        name="qwen2.5",
        reference="Qwen2ForCausalLM",
        build_hf=lambda: Qwen2ForCausalLM(_qwen_config()),
        build_olm=lambda: Qwen2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=QWEN_ROPE_THETA,
            rms_norm_eps=QWEN_RMS_EPS,
        ),
        build_map=lambda olm, hf: llama_family_map(
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            qkv_bias=True,
        ),
        notes="Biases on Q/K/V only. The Q/K biases are permuted with the weights.",
    ),
    ParityCase(
        name="mistral",
        reference="MistralForCausalLM",
        build_hf=lambda: MistralForCausalLM(_mistral_config()),
        build_olm=lambda: MistralSmall3_1_Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=MISTRAL_ROPE_THETA,
            rms_norm_eps=MISTRAL_RMS_EPS,
            tie_weights=False,
        ),
        build_map=lambda olm, hf: llama_family_map(
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            tied_embeddings=False,
        ),
        notes=(
            "Untied output head, so lm_head is mapped as its own parameter. "
            "Attention geometry is the conventional head_dim = embed_dim / "
            "num_heads; see reference_deviations -- the released checkpoint's "
            "head_dim cannot currently be expressed by OLM's preset."
        ),
        reference_deviations=(
            "head_dim is derived as embed_dim / num_heads. The released "
            "Mistral-Small-3.1-24B config sets an explicit head_dim=128 with "
            "hidden_size=5120 and 32 query heads, so q_proj is 4096 wide rather "
            "than 5120. MistralSmall3_1_Model takes no head_dim argument, so its "
            "24B preset derives 160-dimensional heads and cannot reproduce the "
            "released projection shapes. This case therefore validates the block "
            "graph, not the advertised preset's geometry. Tracked upstream; "
            "compare Qwen3Model, which does accept head_dim.",
        ),
    ),
]
