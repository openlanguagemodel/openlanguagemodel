import importlib.util
import inspect
import sys
from pathlib import Path

import olm.models
from olm.models import DeepSeekV3Model, Llama3Model

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gen_model_zoo", ROOT / "scripts" / "gen_model_zoo.py"
)
zoo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = zoo  # dataclasses look their module up here
_spec.loader.exec_module(zoo)


class _TinyLlama(Llama3Model):
    def __init__(self):
        super().__init__(
            vocab_size=64,
            embed_dim=32,
            intermediate_size=48,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=16,
        )


class _TinyMHALlama(Llama3Model):
    def __init__(self):
        super().__init__(
            vocab_size=64,
            embed_dim=32,
            intermediate_size=48,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=16,
        )


class _TinyDeepSeek(DeepSeekV3Model):
    def __init__(self):
        super().__init__(
            vocab_size=64,
            embed_dim=32,
            num_layers=3,
            num_heads=4,
            max_seq_len=16,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=8,
            dense_intermediate_size=48,
            moe_intermediate_size=16,
            num_experts=6,
            num_shared_experts=1,
            top_k=2,
            first_k_dense_replace=1,
            num_mtp_modules=1,
            tie_weights=False,
        )


def _measured(cls, base):
    preset = zoo.Preset(cls=cls, exported=False)
    zoo.measure(preset, base)
    return preset


def _count(model):
    return sum(p.numel() for p in model.parameters())


def test_measure_counts_what_the_preset_builds():
    preset = _measured(_TinyLlama, Llama3Model)

    assert preset.params_total == _count(_TinyLlama())
    assert preset.params_active is None
    assert preset.params_mtp == 0
    assert preset.tied is True
    assert preset.config["num_kv_heads"] == 2
    assert preset.dims == {"layers": 2, "hidden": 32, "context": 16, "vocab": 64}
    assert preset.mixers == [("GQA 4/2", 2)]
    assert preset.ffns == [("SwiGLU", 2)]
    assert list(preset.tags.values()) == [
        "GQA",
        "RoPE",
        "SwiGLU",
        "RMSNorm",
        "tied embeddings",
    ]


def test_measure_counts_active_moe_parameters_without_unused_experts_or_mtp():
    preset = _measured(_TinyDeepSeek, DeepSeekV3Model)
    model = _TinyDeepSeek()
    moe_blocks = [block for block in model.transformer_blocks if block.use_moe]
    expert = _count(moe_blocks[0].ffn.experts[0])
    unused = len(moe_blocks) * (6 - 2) * expert

    assert preset.params_total == _count(model)
    assert preset.params_mtp == _count(model.mtp)
    assert preset.params_active == _count(model) - _count(model.mtp) - unused
    assert preset.tied is False
    assert preset.tags["ffn"] == "MoE (6 SwiGLU experts, top-2, 1 shared)"
    assert preset.tags["layer_split"] == "1 dense + 2 MoE layers"
    assert preset.tags["routing"] == "sigmoid routing with aux-loss-free bias"
    assert preset.tags["mtp"] == "MTP head"


def test_family_tags_merge_presets_per_category():
    family = zoo.Family(cls=Llama3Model, exported=True)
    family.presets = [
        _measured(_TinyMHALlama, Llama3Model),
        _measured(_TinyLlama, Llama3Model),
    ]

    assert zoo.family_tags(family)[0] == "MHA / GQA"
    assert zoo.family_tags(family)[1:] == [
        "RoPE",
        "SwiGLU",
        "RMSNorm",
        "tied embeddings",
    ]


def test_discover_finds_every_exported_preset_and_family():
    families, errors = zoo.discover()
    presets = {p.cls.__name__ for f in families for p in f.presets}
    bases = {f.cls.__name__ for f in families}

    assert errors == []
    for name in olm.models.__all__:
        cls = getattr(olm.models, name)
        own_init = "__init__" in vars(cls)
        takes_no_args = len(inspect.signature(cls.__init__).parameters) == 1
        if own_init and takes_no_args:
            assert name in presets
        else:
            assert name in bases


def test_human_formats_parameter_counts():
    assert zoo.human(None) == "—"
    assert zoo.human(124_439_808) == "124.4M"
    assert zoo.human(7_504_924_672) == "7.50B"
    assert zoo.human(1_026_408_255_488) == "1.03T"
