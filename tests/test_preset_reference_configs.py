"""Check every named model preset against the released config it is named after.

Each fixture in ``tests/fixtures/reference_configs/<Preset>.json`` holds the
fields of one released ``config.json`` (copied verbatim) that the preset has to
reproduce, together with the URL of the exact revision they were read from.
``FAMILIES`` maps every constructor argument of each base model class to the
released field it comes from, so a failure names the argument, the expected
value and where that value was published.

The hyperparameter checks never build a model: the base class ``__init__`` is
patched and the keyword arguments the preset passes are compared, merged with
the base signature's defaults for anything the preset leaves out. The
parameter-count checks build presets on the ``meta`` device, which allocates
no memory.

To add a preset: add its base class to ``FAMILIES`` if it is new, then write
``<Preset>.json`` with ``preset``, ``reference`` (the model page), ``source``
(the config file's URL at a pinned revision) and ``config`` (every field its
``FAMILIES`` entry reads, copied verbatim). Add ``num_parameters`` and
``num_parameters_source`` only when the preset should match a published or
independently computed count exactly and builds on ``meta`` in about a second.
"""

import inspect
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import olm.models

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reference_configs"

_MISSING = object()


class Field:
    """
    A value read from a released config.

    Args:
        paths (str | tuple[str, ...]): Dotted path(s) into the config, e.g.
            ``"text_config.hidden_size"``.
        transform (callable, optional): Maps the value(s) at ``paths`` to the
            OLM argument. Required when more than one path is given.
        default (optional): Value to use for a path that is absent from the
            released file, i.e. the default of the config class that reads it.
    """

    def __init__(self, paths, transform=None, default=_MISSING):
        self.paths = (paths,) if isinstance(paths, str) else tuple(paths)
        self.transform = transform
        self.default = default

    def __call__(self, config):
        values = [_lookup(config, path, self.default) for path in self.paths]
        if self.transform is None:
            return values[0]
        return self.transform(*values)

    def describe(self):
        return " + ".join(self.paths)


def _lookup(config, path, default=_MISSING):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            if default is _MISSING:
                raise KeyError(f"{path!r} is not in the fixture")
            return default
        value = value[key]
    return value


def _tie(prefix="", default=_MISSING):
    # A config.json that omits tie_word_embeddings gets its config class's
    # default; call sites that rely on one name the class.
    return Field(f"{prefix}tie_word_embeddings", default=default)


def _period(layer_types, marker="full_attention"):
    """Period ``p`` such that layer ``i`` is ``marker`` iff ``(i + 1) % p == 0``."""
    period = layer_types.index(marker) + 1
    expected = [(i + 1) % period == 0 for i in range(len(layer_types))]
    actual = [t == marker for t in layer_types]
    assert actual == expected, f"layer schedule is not periodic: {layer_types}"
    return period


def _dense_gqa(
    prefix="",
    eps=True,
    head_dim=False,
    ffn=True,
    tie_default=_MISSING,
    rope_theta_default=_MISSING,
):
    """Fields shared by the Llama-style GQA families."""
    fields = {
        "vocab_size": Field(f"{prefix}vocab_size"),
        "embed_dim": Field(f"{prefix}hidden_size"),
        "num_layers": Field(f"{prefix}num_hidden_layers"),
        "num_heads": Field(f"{prefix}num_attention_heads"),
        "num_kv_heads": Field(f"{prefix}num_key_value_heads"),
        "max_seq_len": Field(f"{prefix}max_position_embeddings"),
        "rope_theta": Field(f"{prefix}rope_theta", float, default=rope_theta_default),
        "tie_weights": _tie(prefix, tie_default),
    }
    if ffn:
        fields["intermediate_size"] = Field(f"{prefix}intermediate_size")
    if eps:
        fields["rms_norm_eps"] = Field(f"{prefix}rms_norm_eps")
    if head_dim:
        fields["head_dim"] = Field(f"{prefix}head_dim")
    return fields


def _mla(prefix=""):
    return {
        "kv_lora_rank": Field(f"{prefix}kv_lora_rank"),
        "qk_nope_head_dim": Field(f"{prefix}qk_nope_head_dim"),
        "qk_rope_head_dim": Field(f"{prefix}qk_rope_head_dim"),
        "v_head_dim": Field(f"{prefix}v_head_dim"),
    }


_DEEPSEEK_V3 = {
    "vocab_size": Field("vocab_size"),
    "embed_dim": Field("hidden_size"),
    "num_layers": Field("num_hidden_layers"),
    "num_heads": Field("num_attention_heads"),
    "max_seq_len": Field("max_position_embeddings"),
    **_mla(),
    "q_lora_rank": Field("q_lora_rank"),
    "dense_intermediate_size": Field("intermediate_size"),
    "moe_intermediate_size": Field("moe_intermediate_size"),
    "num_experts": Field("n_routed_experts"),
    "num_shared_experts": Field("n_shared_experts"),
    "top_k": Field("num_experts_per_tok"),
    "first_k_dense_replace": Field("first_k_dense_replace"),
    "routed_scaling_factor": Field("routed_scaling_factor"),
    "rope_theta": Field("rope_theta", float),
    "rms_norm_eps": Field("rms_norm_eps"),
    "tie_weights": _tie(),
}


# Base model class -> {constructor argument: released field}. Every argument of
# the base class is listed here or in NOT_FROM_CONFIG
# (test_families_cover_every_constructor_argument enforces it).
FAMILIES = {
    "GPT2Model": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("n_embd"),
        "num_layers": Field("n_layer"),
        "num_heads": Field("n_head"),
        "max_seq_len": Field("n_positions"),
        "dropout": Field("resid_pdrop"),
        "tie_weights": _tie(default=True),  # GPT2Config's default
    },
    "OPTModel": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field("ffn_dim"),
        "num_layers": Field("num_hidden_layers"),
        "num_heads": Field("num_attention_heads"),
        "dropout": Field("dropout"),
        "tie_weights": _tie(default=True),  # OPTConfig's default
    },
    # The Llama 2 configs predate rope_theta; LlamaConfig defaults it to 10000.
    "Llama2Model": _dense_gqa(eps=False, rope_theta_default=10000.0),
    "Llama3Model": _dense_gqa(eps=False),
    "Qwen2Model": _dense_gqa(),
    "Nanbeige4_1_Model": _dense_gqa(head_dim=True),
    "Phi3Model": {
        # Phi-3.5-mini uses the Phi3Config keys, Phi-3-small its own
        # remote-code names; exactly one of each pair is present.
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field(
            ("intermediate_size", "ff_intermediate_size"),
            lambda a, b: a if a is not None else b,
            default=None,
        ),
        "num_layers": Field("num_hidden_layers"),
        "num_heads": Field("num_attention_heads"),
        "num_kv_heads": Field("num_key_value_heads"),
        "max_seq_len": Field("max_position_embeddings"),
        "rope_theta": Field(
            ("rope_theta", "rope_embedding_base"),
            lambda a, b: float(a if a is not None else b),
            default=None,
        ),
        "activation": Field(
            "hidden_act", lambda act: {"silu": "swiglu", "gegelu": "geglu"}[act]
        ),
        # Phi-3-small omits the key. Its remote-code config keeps the
        # transformers 4.x PretrainedConfig default (True), and its model
        # declares _tied_weights_keys = ["lm_head.weight"].
        "tie_weights": _tie(default=True),
    },
    "Phi4Model": _dense_gqa(eps=False),
    "OLMoModel": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field("intermediate_size"),
        "num_layers": Field("num_hidden_layers"),
        "num_heads": Field("num_attention_heads"),
        "max_seq_len": Field("max_position_embeddings"),
        "tie_weights": _tie(),
    },
    "Olmo3Model": {
        **_dense_gqa(),
        "head_dim": Field(("hidden_size", "num_attention_heads"), lambda d, h: d // h),
        "sliding_window": Field("sliding_window"),
        "sliding_window_pattern": Field("layer_types", _period),
    },
    "Gemma2Model": {
        # Gemma2Config ties by default; the released configs omit the key.
        **_dense_gqa(eps=False, head_dim=True, tie_default=True),
        "sliding_window": Field("sliding_window"),
        "attn_logit_softcap": Field("attn_logit_softcapping"),
        "final_logit_softcap": Field("final_logit_softcapping"),
        "query_pre_attn_scalar": Field("query_pre_attn_scalar", float),
    },
    "Gemma3Model": {
        **{
            k: v
            # Gemma3TextConfig ties by default; the released configs omit it.
            for k, v in _dense_gqa(
                "text_config.", head_dim=True, tie_default=True
            ).items()
            if k != "rope_theta"
        },
        "local_rope_theta": Field("text_config.rope_local_base_freq"),
        "global_rope_theta": Field("text_config.rope_theta"),
        "sliding_window": Field("text_config.sliding_window"),
        "sliding_window_pattern": Field("text_config.sliding_window_pattern"),
        "query_pre_attn_scalar": Field("text_config.query_pre_attn_scalar", float),
    },
    # MistralConfig defaults tie_word_embeddings to False; the text_config
    # does not set it.
    "MistralSmall3_1_Model": _dense_gqa(
        "text_config.", head_dim=True, tie_default=False
    ),
    "Qwen3Model": {
        **_dense_gqa(head_dim=True, ffn=False),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "num_experts": Field("num_experts"),
        "top_k": Field("num_experts_per_tok"),
    },
    "Qwen3NextModel": {
        **_dense_gqa(head_dim=True, ffn=False),
        "linear_num_key_heads": Field("linear_num_key_heads"),
        "linear_num_value_heads": Field("linear_num_value_heads"),
        "linear_key_head_dim": Field("linear_key_head_dim"),
        "linear_value_head_dim": Field("linear_value_head_dim"),
        "linear_conv_kernel_size": Field("linear_conv_kernel_dim"),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "num_experts": Field("num_experts"),
        "num_shared_experts": Field(
            "shared_expert_intermediate_size", lambda width: int(width > 0)
        ),
        "top_k": Field("num_experts_per_tok"),
        "full_attention_interval": Field("full_attention_interval"),
        "partial_rotary_factor": Field("partial_rotary_factor"),
    },
    "Qwen3_5_Model": {
        "vocab_size": Field("text_config.vocab_size"),
        "embed_dim": Field("text_config.hidden_size"),
        "num_layers": Field("text_config.num_hidden_layers"),
        "full_attn_heads": Field("text_config.num_attention_heads"),
        "full_attn_kv_heads": Field("text_config.num_key_value_heads"),
        "full_attn_head_dim": Field("text_config.head_dim"),
        "linear_num_key_heads": Field("text_config.linear_num_key_heads"),
        "linear_num_value_heads": Field("text_config.linear_num_value_heads"),
        "linear_key_head_dim": Field("text_config.linear_key_head_dim"),
        "linear_value_head_dim": Field("text_config.linear_value_head_dim"),
        "linear_conv_kernel_dim": Field("text_config.linear_conv_kernel_dim"),
        "max_seq_len": Field("text_config.max_position_embeddings"),
        "moe_intermediate_size": Field("text_config.moe_intermediate_size"),
        "shared_expert_intermediate_size": Field(
            "text_config.shared_expert_intermediate_size"
        ),
        "num_experts": Field("text_config.num_experts"),
        "num_experts_per_tok": Field("text_config.num_experts_per_tok"),
        "num_mtp_heads": Field("text_config.mtp_num_hidden_layers"),
        "rope_theta": Field("text_config.rope_parameters.rope_theta", float),
        "partial_rotary_factor": Field(
            "text_config.rope_parameters.partial_rotary_factor"
        ),
        "rms_norm_eps": Field("text_config.rms_norm_eps"),
        "tie_weights": _tie(),
    },
    "Llama4MaverickModel": {
        "vocab_size": Field("text_config.vocab_size"),
        "embed_dim": Field("text_config.hidden_size"),
        "intermediate_size": Field("text_config.intermediate_size_mlp"),
        "moe_intermediate_size": Field("text_config.intermediate_size"),
        "num_layers": Field("text_config.num_hidden_layers"),
        "num_heads": Field("text_config.num_attention_heads"),
        "num_kv_heads": Field("text_config.num_key_value_heads"),
        "head_dim": Field("text_config.head_dim"),
        "max_seq_len": Field("text_config.max_position_embeddings"),
        "chunk_size": Field("text_config.attention_chunk_size"),
        "num_experts": Field("text_config.num_local_experts"),
        "top_k": Field("text_config.num_experts_per_tok"),
        "interleave_moe_layer_step": Field("text_config.interleave_moe_layer_step"),
        # no_rope_layers[i] == 0 marks the global NoPE layers.
        "nope_layer_interval": Field(
            "text_config.no_rope_layers",
            lambda flags: _period(
                ["nope" if f == 0 else "rope" for f in flags], "nope"
            ),
        ),
        "rope_theta": Field("text_config.rope_theta"),
        "rms_norm_eps": Field("text_config.rms_norm_eps"),
        # Llama4TextConfig defaults tie_word_embeddings to False.
        "tie_weights": _tie("text_config.", default=False),
    },
    "DeepSeekV3Model": {
        **_DEEPSEEK_V3,
        "num_mtp_modules": Field("num_nextn_predict_layers"),
    },
    "KimiK2Model": _DEEPSEEK_V3,
    "MistralLarge3Model": {
        # params.json (Mistral's native format), not a transformers config.
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("dim"),
        "num_layers": Field("n_layers"),
        "num_heads": Field("n_heads"),
        "max_seq_len": Field("max_position_embeddings"),
        **_mla(),
        "q_lora_rank": Field("q_lora_rank"),
        "dense_intermediate_size": Field("hidden_dim"),
        "moe_intermediate_size": Field("moe.expert_hidden_dim"),
        "num_experts": Field("moe.num_experts"),
        "num_shared_experts": Field("moe.num_shared_experts"),
        "top_k": Field("moe.num_experts_per_tok"),
        "first_k_dense_replace": Field("moe.first_k_dense_replace"),
        "rope_theta": Field("rope_theta"),
        "rms_norm_eps": Field("norm_eps"),
        "tie_weights": Field("tied_embeddings"),
    },
    "KimiLinearModel": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "num_layers": Field("num_hidden_layers"),
        "kda_num_heads": Field("linear_attn_config.num_heads"),
        "kda_head_dim": Field("linear_attn_config.head_dim"),
        "kda_conv_kernel_size": Field("linear_attn_config.short_conv_kernel_size"),
        "mla_num_heads": Field("num_attention_heads"),
        "max_seq_len": Field("model_max_length"),
        **_mla(),
        "q_lora_rank": Field("q_lora_rank"),
        "intermediate_size": Field("intermediate_size"),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "num_experts": Field("num_experts"),
        "num_shared_experts": Field("num_shared_experts"),
        "top_k": Field("num_experts_per_token"),
        # MLA on every 4th layer and on the last one: 1-indexed layers
        # 4, 8, ..., 24 and 27 in linear_attn_config.full_attn_layers.
        "full_attention_interval": Field(
            ("linear_attn_config.full_attn_layers", "num_hidden_layers"),
            lambda layers, n: _period(
                ["full" if i + 1 in layers else "kda" for i in range(n - 1)], "full"
            ),
        ),
        "first_k_dense_replace": Field("first_k_dense_replace"),
        "rope_theta": Field("rope_theta"),
        "routed_scaling_factor": Field("routed_scaling_factor"),
        "rms_norm_eps": Field("rms_norm_eps"),
        "tie_weights": _tie(),
    },
    "MiniMaxM2Model": {
        **_dense_gqa(head_dim=True, ffn=False),
        "moe_intermediate_size": Field("intermediate_size"),
        "num_experts": Field("num_local_experts"),
        "top_k": Field("num_experts_per_tok"),
        "rotary_percentage": Field(
            ("rotary_dim", "head_dim"), lambda rotary, head: rotary / head
        ),
        "use_qk_norm": Field("use_qk_norm"),
    },
    "MiniMaxM2_5_Model": {
        **_dense_gqa(head_dim=True, ffn=False),
        "moe_intermediate_size": Field("intermediate_size"),
        "num_experts": Field("num_local_experts"),
        "num_experts_per_tok": Field("num_experts_per_tok"),
        "num_mtp_modules": Field("num_mtp_modules"),
        "partial_rotary_factor": Field(
            ("rotary_dim", "head_dim"), lambda rotary, head: rotary / head
        ),
    },
    "Ling2_5_Model": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field("intermediate_size"),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "num_layers": Field("num_hidden_layers"),
        "num_heads": Field("num_attention_heads"),
        "head_dim": Field("head_dim"),
        **_mla(),
        "q_lora_rank": Field("q_lora_rank"),
        "max_seq_len": Field("max_position_embeddings"),
        "num_experts": Field("num_experts"),
        "num_experts_per_tok": Field("num_experts_per_tok"),
        "layer_group_size": Field("layer_group_size"),
        "first_k_dense": Field("first_k_dense_replace"),
        "rope_theta": Field("rope_theta", float),
        "rms_norm_eps": Field("rms_norm_eps"),
        "routed_scaling_factor": Field("routed_scaling_factor"),
        "tie_weights": _tie(),
    },
    "Sarvam30B_Model": {
        **_dense_gqa(head_dim=True),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "shared_expert_intermediate_size": Field("moe_shared_expert_intermediate_size"),
        "num_experts": Field("num_experts"),
        "num_experts_per_tok": Field("num_experts_per_tok"),
        "first_k_dense": Field("first_k_dense_replace"),
        "routed_scaling_factor": Field("routed_scaling_factor"),
    },
    "Sarvam105B_Model": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field("intermediate_size"),
        "moe_intermediate_size": Field("moe_intermediate_size"),
        "num_layers": Field("num_hidden_layers"),
        "num_heads": Field("num_attention_heads"),
        **_mla(),
        "max_seq_len": Field("max_position_embeddings"),
        "num_experts": Field("num_experts"),
        "num_experts_per_tok": Field("num_experts_per_tok"),
        "first_k_dense": Field("first_k_dense_replace"),
        "rope_theta": Field("rope_theta"),
        "rms_norm_eps": Field("rms_norm_eps"),
        "routed_scaling_factor": Field("routed_scaling_factor"),
        "tie_weights": _tie(),
    },
    "Step3_5_Flash_Model": {
        "vocab_size": Field("vocab_size"),
        "embed_dim": Field("hidden_size"),
        "intermediate_size": Field("intermediate_size"),
        # The shared expert has the routed experts' width (share_expert_dim).
        "moe_intermediate_size": Field(
            ("moe_intermediate_size", "share_expert_dim"),
            lambda routed, shared: routed if routed == shared else (routed, shared),
        ),
        "num_layers": Field("num_hidden_layers"),
        "full_attn_heads": Field("num_attention_heads"),
        "sliding_attn_heads": Field("attention_other_setting.num_attention_heads"),
        "num_kv_heads": Field(
            ("num_attention_groups", "attention_other_setting.num_attention_groups"),
            lambda full, sliding: full if full == sliding else (full, sliding),
        ),
        "head_dim": Field("head_dim"),
        "max_seq_len": Field("max_position_embeddings"),
        "window_size": Field("sliding_window"),
        "num_experts": Field("moe_num_experts"),
        "num_experts_per_tok": Field("moe_top_k"),
        "num_mtp_heads": Field("num_nextn_predict_layers"),
        "routed_scaling_factor": Field("moe_router_scaling_factor"),
        # Absent from config.json; Step3p5Config defaults it to 1e-5.
        "rms_norm_eps": Field("rms_norm_eps", default=1e-5),
        "tie_weights": _tie(),
        # The per-layer lists also cover the MTP layers stored after the
        # decoder stack; the decoder uses the first num_hidden_layers entries.
        "layer_types": Field(
            ("layer_types", "num_hidden_layers"),
            lambda types, n: [t.replace("_attention", "") for t in types[:n]],
        ),
        "moe_layers": Field(
            "moe_layers_enum", lambda enum: {int(i) for i in enum.split(",")}
        ),
        "rope_thetas": Field(
            ("rope_theta", "num_hidden_layers"), lambda thetas, n: thetas[:n]
        ),
        "partial_rotary_factors": Field(
            ("partial_rotary_factors", "num_hidden_layers"),
            lambda factors, n: factors[:n],
        ),
    },
}

# Constructor arguments that no released config sets.
NOT_FROM_CONFIG = {
    "dropout": "a training-time setting rather than part of the architecture",
}


def _fixtures():
    return sorted(FIXTURE_DIR.glob("*.json"))


def _load(path):
    return json.loads(path.read_text())


def _preset_names():
    """Named presets exported by ``olm.models``: classes built with no arguments."""
    names = []
    for name in olm.models.__all__:
        cls = getattr(olm.models, name)
        if not inspect.signature(cls).parameters:
            names.append(name)
    return sorted(names)


def _base(preset_cls):
    return preset_cls.__mro__[1]


def _preset_arguments(preset_cls):
    """Every constructor argument the preset gives its base class."""
    base = _base(preset_cls)
    signature = inspect.signature(base.__init__)
    with patch.object(base, "__init__", return_value=None) as init:
        preset_cls()
    bound = signature.bind(None, *init.call_args.args, **init.call_args.kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop(next(iter(signature.parameters)))  # self
    return arguments


def test_every_preset_has_a_reference_fixture():
    assert [path.stem for path in _fixtures()] == _preset_names()


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda path: path.stem)
def test_fixture_names_its_preset_and_a_pinned_source(fixture):
    data = _load(fixture)

    assert data["preset"] == fixture.stem
    assert data["reference"].startswith("https://huggingface.co/")
    # A revision hash, not a branch, so the values cannot drift.
    assert re.fullmatch(
        r"https://huggingface\.co/[\w.-]+/[\w.-]+/blob/[0-9a-f]{40}/\w+\.json",
        data["source"],
    ), data["source"]


@pytest.mark.parametrize("base_name", sorted(FAMILIES))
def test_families_cover_every_constructor_argument(base_name):
    base = getattr(olm.models, base_name)
    parameters = list(inspect.signature(base.__init__).parameters)[1:]
    covered = set(FAMILIES[base_name])
    missing = set(parameters) - covered - set(NOT_FROM_CONFIG)

    assert not missing, f"{base_name} arguments with no released field: {missing}"
    assert covered <= set(parameters), covered - set(parameters)


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda path: path.stem)
def test_preset_matches_its_released_config(fixture):
    data = _load(fixture)
    preset_cls = getattr(olm.models, fixture.stem)
    arguments = _preset_arguments(preset_cls)
    fields = FAMILIES[_base(preset_cls).__name__]

    mismatches = []
    for argument, field in fields.items():
        expected = field(data["config"])
        if arguments[argument] != expected:
            mismatches.append(
                f"{argument}={arguments[argument]!r}, released "
                f"{field.describe()} gives {expected!r}"
            )
    assert not mismatches, f"{fixture.stem} vs {data['source']}:\n" + "\n".join(
        mismatches
    )


def _counted_fixtures():
    return [path for path in _fixtures() if "num_parameters" in _load(path)]


@pytest.mark.parametrize("fixture", _counted_fixtures(), ids=lambda path: path.stem)
def test_preset_parameter_count_matches_the_reference(fixture):
    data = _load(fixture)
    with torch.device("meta"):
        model = getattr(olm.models, fixture.stem)()
    num_parameters = sum(p.numel() for p in model.parameters())

    assert num_parameters == data["num_parameters"], data["num_parameters_source"]


def test_qwen3_5_layer_schedule_matches_the_released_layer_types():
    from olm.models.alibaba.qwen3_5 import QWEN3_5_LAYER_TYPES

    config = _load(FIXTURE_DIR / "Qwen3_5_397B.json")["config"]["text_config"]
    num_layers = config["num_hidden_layers"]

    assert QWEN3_5_LAYER_TYPES[:num_layers] == config["layer_types"]


def test_kimi_linear_final_layer_is_mla_as_released():
    config = _load(FIXTURE_DIR / "KimiLinear48BA3B.json")["config"]
    full_attn_layers = config["linear_attn_config"]["full_attn_layers"]

    # KimiLinearModel always makes the last layer MLA; the release agrees.
    assert config["num_hidden_layers"] in full_attn_layers


def test_llama4_moe_layers_follow_the_interleave_step():
    config = _load(FIXTURE_DIR / "Llama4Maverick_400B.json")["config"]["text_config"]
    step = config["interleave_moe_layer_step"]
    num_layers = config["num_hidden_layers"]

    # Llama4MaverickModel: use_moe iff layer_idx % step == step - 1.
    assert config["moe_layers"] == [
        i for i in range(num_layers) if i % step == step - 1
    ]
