#!/usr/bin/env python3
"""Generate the OLM Model Zoo: ``docs/models/index.md`` and ``docs/models/zoo.json``.

Everything on the generated page is read from the code, so the page never needs a
hand edit:

- Families and presets are discovered by walking ``olm.models``. A preset is a
  class whose ``__init__`` takes no arguments; its family is the base class it
  calls ``super().__init__`` on.
- Each preset's configuration is the keyword arguments it passes to that base
  class, captured with ``unittest.mock`` so nothing is allocated.
- Parameter counts come from building each preset under ``torch.device("meta")``
  (shapes only, no memory) and counting the parameters of the real module tree.
- Architecture tags (attention type, feed-forward type, MoE routing, norms,
  positions) come from the module classes and attributes found in that tree.

Run it from the repository root after adding or changing a model family::

    PYTHONPATH=src python scripts/gen_model_zoo.py          # rewrite both files
    PYTHONPATH=src python scripts/gen_model_zoo.py --check  # exit 1 if stale

Building every preset takes about a minute and a half on one CPU core and peaks at
about 1.1 GB of RAM, most of it spent constructing the expert modules of the
largest MoE models.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import inspect
import json
import pkgutil
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODELS_DOCS = ROOT / "docs" / "models"
MARKDOWN_OUT = MODELS_DOCS / "index.md"
JSON_OUT = MODELS_DOCS / "zoo.json"
REPO_URL = "https://github.com/openlanguagemodel/openlanguagemodel"
BLOB_URL = f"{REPO_URL}/blob/main"
REGENERATE = "PYTHONPATH=src python scripts/gen_model_zoo.py"

# An architecture guide opts in to the zoo with a comment naming the family base
# classes it covers, e.g. ``<!-- model-zoo: KimiK2Model KimiLinearModel -->``.
GUIDE_MARKER = re.compile(r"<!--\s*model-zoo:\s*([\w\s]+?)\s*-->")

# Display names. A family missing here falls back to a name derived from its
# class (``MistralSmall3_1_Model`` -> ``MistralSmall 3.1``), so new families
# still render; add an entry to give them their official spelling.
FAMILY_NAMES = {
    "DeepSeekV3Model": "DeepSeek-V3",
    "Gemma2Model": "Gemma 2",
    "Gemma3Model": "Gemma 3",
    "GPT2Model": "GPT-2",
    "KimiK2Model": "Kimi K2",
    "KimiLinearModel": "Kimi Linear",
    "Ling2_5_Model": "Ling 2.5",
    "Llama2Model": "Llama 2",
    "Llama3Model": "Llama 3.1 / 3.2",
    "Llama4MaverickModel": "Llama 4 Maverick",
    "MiniMaxM2Model": "MiniMax-M2",
    "MiniMaxM2_5_Model": "MiniMax-M2.5",
    "MistralLarge3Model": "Mistral Large 3",
    "MistralSmall3_1_Model": "Mistral Small 3.1",
    "Nanbeige4_1_Model": "Nanbeige 4.1",
    "OLMoModel": "OLMo",
    "Olmo3Model": "OLMo 3",
    "OPTModel": "OPT",
    "Phi3Model": "Phi-3 / Phi-3.5",
    "Phi4Model": "Phi-4",
    "Qwen2Model": "Qwen 2.5",
    "Qwen3Model": "Qwen3 MoE",
    "Qwen3_5_Model": "Qwen3.5",
    "Qwen3NextModel": "Qwen3-Next",
    "Sarvam30B_Model": "Sarvam 30B",
    "Sarvam105B_Model": "Sarvam 105B",
    "Step3_5_Flash_Model": "Step 3.5 Flash",
}

# Organisation shown for each vendor subpackage of ``olm.models``.
ORGANISATIONS = {
    "alibaba": "Alibaba (Qwen)",
    "allenai": "Ai2",
    "deepseekai": "DeepSeek",
    "facebook": "Meta",
    "google": "Google",
    "inclusionai": "inclusionAI",
    "meta": "Meta",
    "microsoft": "Microsoft",
    "minimax": "MiniMax",
    "mistralai": "Mistral AI",
    "moonshotai": "Moonshot AI",
    "nanbeige": "Nanbeige",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "sarvam": "Sarvam AI",
    "stepfun": "StepFun",
}

# Short labels for token mixers (attention and its linear-time replacements).
# Grouped-query modules are labelled MHA / GQA / MQA from their head counts.
MIXER_LABELS = {
    "FlashAttention": "MHA",
    "FlashAttentionwithRoPE": "MHA",
    "MultiHeadAttention": "MHA",
    "MultiHeadAttentionwithRoPE": "MHA",
    "MultiHeadAttentionwithALiBi": "MHA (ALiBi)",
    "MultiHeadLatentAttention": "MLA",
    "GatedAttention": "gated attention",
    "GatedDeltaNet": "Gated DeltaNet",
    "KimiDeltaAttention": "Kimi Delta Attention",
    "LightningAttention": "Lightning Attention",
    "SparseAttention": "block-sparse attention",
    "SparseAttentionwithRoPE": "block-sparse attention",
}

FFN_LABELS = {
    "SwiGLUFFN": "SwiGLU",
    "GeGLUFFN": "GeGLU",
}

# Module classes that are parts of a mixer rather than mixers themselves.
NOT_MIXERS = {"HeadGate"}

# Order of the architecture tags within a row. A family merges its presets' tags
# per category, so a family whose presets use MHA and GQA reads "MHA / GQA".
TAG_CATEGORIES = (
    "mixer",
    "qk_norm",
    "head_gate",
    "attn_softcap",
    "positions",
    "nope",
    "ffn",
    "layer_split",
    "routing",
    "norm",
    "final_softcap",
    "mtp",
    "embeddings",
)

# Constructor arguments that give a preset's headline dimensions, in order of
# preference. Families that name them differently still fill the table because
# the generator falls back to reading the built modules.
DIM_KEYS = {
    "layers": ("num_layers", "num_hidden_layers", "n_layers"),
    "hidden": ("embed_dim", "hidden_size", "d_model"),
    "context": ("max_seq_len", "max_position_embeddings", "context_length"),
    "vocab": ("vocab_size",),
}


@dataclass
class Preset:
    """One named preset and everything measured from it."""

    cls: type
    exported: bool
    config: dict = field(default_factory=dict)
    params_total: int = 0
    params_active: int | None = None
    params_mtp: int = 0
    tied: bool | None = None
    mixers: list[tuple[str, int]] = field(default_factory=list)
    ffns: list[tuple[str, int]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    dims: dict[str, int | None] = field(default_factory=dict)


@dataclass
class Family:
    """A configurable base class and its presets."""

    cls: type
    exported: bool
    presets: list[Preset] = field(default_factory=list)
    guide: str | None = None


def is_preset(cls: type) -> bool:
    """A preset defines its own zero-argument ``__init__`` on top of a base model."""
    if "__init__" not in vars(cls):
        return False
    params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    if params:
        return False
    parent = cls.__mro__[1]
    return parent.__module__.startswith("olm.models.") and takes_arguments(parent)


def takes_arguments(cls: type) -> bool:
    params = inspect.signature(cls.__init__).parameters
    return any(name != "self" for name in params)


def discover() -> tuple[list[Family], list[str]]:
    """Import every ``olm.models`` module and group presets under their base."""
    import olm.models

    exported = set(olm.models.__all__)
    errors: list[str] = []
    classes: dict[str, type] = {}
    for info in pkgutil.walk_packages(olm.models.__path__, "olm.models."):
        try:
            module = import_module(info.name)
        except Exception as exc:  # pragma: no cover - reported, not fatal
            errors.append(f"{info.name}: {type(exc).__name__}: {exc}")
            continue
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and issubclass(obj, nn.Module)
            ):
                classes[f"{obj.__module__}.{obj.__qualname__}"] = obj

    families: dict[type, Family] = {}
    for cls in classes.values():
        if is_preset(cls):
            base = cls.__mro__[1]
            family = families.setdefault(
                base, Family(cls=base, exported=base.__name__ in exported)
            )
            family.presets.append(Preset(cls=cls, exported=cls.__name__ in exported))
        elif cls.__name__ in exported and takes_arguments(cls):
            families.setdefault(cls, Family(cls=cls, exported=True))

    for family in families.values():
        family.presets.sort(key=lambda preset: source_line(preset.cls))
    ordered = sorted(
        families.values(),
        key=lambda f: (
            organisation(f.cls).casefold(),
            source_path(f.cls),
            source_line(f.cls),
        ),
    )
    return ordered, errors


def source_path(obj: Any) -> str:
    return Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()


def source_line(obj: Any) -> int:
    return inspect.getsourcelines(obj)[1]


def vendor(cls: type) -> str:
    return cls.__module__.split(".")[2]


def organisation(cls: type) -> str:
    return ORGANISATIONS.get(vendor(cls), vendor(cls))


def family_name(cls: type) -> str:
    if cls.__name__ in FAMILY_NAMES:
        return FAMILY_NAMES[cls.__name__]
    name = re.sub(r"_?Model$", "", cls.__name__)
    name = re.sub(r"(\d)_(\d)", r"\1.\2", name)
    name = re.sub(r"(?<=[a-z])(?=\d)", " ", name)
    return name.replace("_", " ")


def docstring_section(cls: type, section: str) -> str | None:
    """Return one ``Section:`` block of a Google-style docstring, dedented."""
    doc = inspect.getdoc(cls) or ""
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"{section}:":
            body = []
            for item in lines[index + 1 :]:
                if item and not item.startswith(" "):
                    break
                body.append(item.strip())
            return " ".join(part for part in body if part) or None
    return None


def capture_config(preset: type, base: type) -> dict:
    """Resolve the arguments a preset passes to its base, without building it."""
    signature = inspect.signature(base.__init__)
    with mock.patch.object(base, "__init__", return_value=None) as init:
        preset()
    bound = signature.bind(None, *init.call_args.args, **init.call_args.kwargs)
    bound.apply_defaults()
    config = dict(bound.arguments)
    config.pop("self", None)
    return {name: jsonable(value) for name, value in config.items()}


def jsonable(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def is_mixer(module: nn.Module) -> bool:
    cls = type(module)
    if cls.__name__ in NOT_MIXERS:
        return False
    return cls.__module__.startswith("olm.nn.attention.") or cls.__name__.endswith(
        ("Attention", "Mixer")
    )


def is_ffn(module: nn.Module) -> bool:
    # A family may wrap its MoE in a class of its own; the experts and top_k
    # attributes identify it wherever it is defined.
    if is_moe(module):
        return True
    return type(module).__module__.startswith(("olm.nn.feedforward.", "olm.nn.moe."))


def is_moe(module: nn.Module) -> bool:
    return isinstance(getattr(module, "experts", None), nn.ModuleList) and isinstance(
        getattr(module, "top_k", None), int
    )


def is_mtp(module: nn.Module) -> bool:
    return "MultiTokenPrediction" in type(module).__name__


def outermost(model: nn.Module, predicate, skip=is_mtp) -> list[nn.Module]:
    """Modules matching ``predicate`` that are not nested in another match.

    Subtrees for which ``skip`` is true (by default the multi-token-prediction
    head, which the default ``forward`` never runs) are not searched.
    """
    found: list[nn.Module] = []

    def visit(module: nn.Module) -> None:
        if module is not model and skip(module):
            return
        if predicate(module):
            found.append(module)
            return
        for child in module.children():
            visit(child)

    visit(model)
    return found


def block_windows(model: nn.Module) -> dict[int, int]:
    """Sliding windows that enclosing blocks set for their mixers, by mixer id.

    Some families keep the attention module unchanged and mask its scores in the
    block instead, with an int ``sliding_window`` attribute that the model sets
    per layer (``None`` on the global layers). The nearest enclosing module with
    that attribute decides.
    """
    windows: dict[int, int] = {}

    def visit(module: nn.Module, window: int | None) -> None:
        if hasattr(module, "sliding_window"):
            value = module.sliding_window
            window = value if type(value) is int else None
        if is_mixer(module):
            if window is not None:
                windows[id(module)] = window
            return
        for child in module.children():
            visit(child, window)

    visit(model, None)
    return windows


def mixer_label(module: nn.Module, window: int | None = None) -> str:
    """Label a token mixer, e.g. ``sliding-window GQA (4096) 32/16``.

    Args:
        module: The attention module or linear-time mixer.
        window: Sliding window applied by the enclosing block, if any. A
            ``window_size`` on the module itself takes precedence.
    """
    name = type(module).__name__
    heads = getattr(module, "num_heads", None)
    kv_heads = getattr(module, "num_kv_heads", None)
    if isinstance(heads, int) and isinstance(kv_heads, int):
        kind = "MHA" if kv_heads == heads else "MQA" if kv_heads == 1 else "GQA"
        label = f"gated {kind}" if name == "GatedAttention" else kind
    else:
        label = MIXER_LABELS.get(name, name)
        if isinstance(heads, int) and label.startswith("MHA"):
            # Plain multi-head attention has one key-value head per query head.
            kv_heads = heads
    own_window = getattr(module, "window_size", None)
    if isinstance(own_window, int):
        window = own_window
    if window is not None:
        label = f"sliding-window {label} ({window})"
    if isinstance(heads, int) and isinstance(kv_heads, int):
        return f"{label} {heads}/{kv_heads}"
    if isinstance(heads, int):
        return f"{label} {heads}h"
    return label


def ffn_label(module: nn.Module) -> str:
    if is_moe(module):
        experts = len(module.experts)
        shared = getattr(module, "shared_experts", None)
        shared_count = len(shared) if isinstance(shared, nn.ModuleList) else 0
        label = f"MoE {experts}e top-{module.top_k}"
        if shared_count:
            label += f" +{shared_count} shared"
        return label
    name = type(module).__name__
    if name in FFN_LABELS:
        return FFN_LABELS[name]
    act = getattr(module, "act", None)
    if act is not None:
        return f"{type(act).__name__} MLP"
    return name


def router_scoring(router: nn.Module) -> str | None:
    """Scoring function of a router, read from its attributes or its source."""
    scoring = getattr(router, "scoring_func", None)
    if scoring:
        return scoring
    try:
        source = inspect.getsource(type(router).forward)
    except (OSError, TypeError):
        return None
    if "sigmoid" in source:
        return "sigmoid"
    if "softmax" in source:
        return "softmax"
    return None


def summarise(counter: Counter) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def measure(preset: Preset, base: type) -> None:
    """Fill a preset's config, parameter counts, layer mix and tags."""
    try:
        preset.config = capture_config(preset.cls, base)
    except Exception as exc:  # pragma: no cover - reported in the JSON
        preset.config = {"error": f"{type(exc).__name__}: {exc}"}

    with torch.device("meta"):
        model = preset.cls()

    preset.params_total = count(model)
    mtp_modules = outermost(model, is_mtp, skip=lambda m: False)
    preset.params_mtp = sum(count(m) for m in mtp_modules)

    mixers = outermost(model, is_mixer)
    windows = block_windows(model)
    labels = [mixer_label(m, windows.get(id(m))) for m in mixers]
    ffns = outermost(model, is_ffn)
    moes = [m for m in ffns if is_moe(m)]
    if moes:
        inactive = sum((len(m.experts) - m.top_k) * count(m.experts[0]) for m in moes)
        preset.params_active = preset.params_total - inactive - preset.params_mtp

    heads = [m for m in model.modules() if type(m).__name__ == "OutputHead"]
    if heads:
        preset.tied = type(heads[-1].projection).__name__ == "_TiedEmbeddingProjection"

    preset.mixers = summarise(Counter(labels))
    preset.ffns = summarise(Counter(ffn_label(m) for m in ffns))
    preset.tags = architecture_tags(model, mixers, labels, ffns, moes, preset)
    preset.dims = dimensions(model, mixers, preset.config)

    del model
    gc.collect()


def architecture_tags(
    model, mixers, labels, ffns, moes, preset: Preset
) -> dict[str, str]:
    """Short, code-derived description of what the architecture is made of.

    ``labels`` are the ``mixer_label`` of each module in ``mixers``. Returns one
    tag per category in ``TAG_CATEGORIES`` that applies.
    """
    tags: dict[str, str] = {}

    kinds = []
    for label in labels:
        kind = re.sub(r" \d+(/\d+|h)$", "", label)  # drop the head counts
        kind = re.sub(r" \(\d+\)$", "", kind)  # and the window size
        if kind not in kinds:
            kinds.append(kind)
    if kinds:
        tags["mixer"] = " + ".join(kinds)

    if any(getattr(m, "q_norm", None) is not None for m in mixers):
        tags["qk_norm"] = "QK-norm"
    if any(type(m).__name__ == "HeadGate" for m in model.modules()):
        tags["head_gate"] = "per-head attention gate"
    if any(getattr(m, "attn_logit_softcap", None) for m in mixers):
        tags["attn_softcap"] = "attention-logit softcap"

    positional = {type(m).__name__ for m in model.modules()}
    if "AbsolutePositionalEmbedding" in positional:
        tags["positions"] = "learned positions"
    elif "PartialRotaryPositionalEmbedding" in positional:
        tags["positions"] = "partial RoPE"
    elif "RotaryPositionalEmbedding" in positional:
        tags["positions"] = "RoPE"
    nope = sum(
        1
        for m in mixers
        if hasattr(m, "rope") and getattr(m, "use_rope", True) is False
    )
    if nope:
        tags["nope"] = f"NoPE on {nope} layers"

    dense = [m for m in ffns if not is_moe(m)]
    if moes:
        first = moes[0]
        expert = ffn_label(first.experts[0])
        shared = getattr(first, "shared_experts", None)
        shared_count = len(shared) if isinstance(shared, nn.ModuleList) else 0
        moe = f"MoE ({len(first.experts)} {expert} experts, top-{first.top_k}"
        moe += f", {shared_count} shared)" if shared_count else ")"
        tags["ffn"] = moe
        if dense:
            tags["layer_split"] = f"{len(dense)} dense + {len(moes)} MoE layers"
        router = getattr(first, "router", None)
        scoring = router_scoring(router) if router is not None else None
        if scoring:
            routing = f"{scoring} routing"
            if getattr(router, "routing_method", None) == "noaux_tc":
                routing += " with aux-loss-free bias"
            tags["routing"] = routing
    elif dense:
        tags["ffn"] = " + ".join(sorted({ffn_label(m) for m in dense}))

    norms = sorted(
        {
            type(m).__name__
            for m in model.modules()
            if type(m).__module__.startswith("olm.nn.norms.")
        }
    )
    if norms:
        tags["norm"] = " + ".join(norms)
    if any("Softcap" in type(m).__name__ for m in model.modules()):
        tags["final_softcap"] = "final-logit softcap"
    if preset.params_mtp:
        tags["mtp"] = "MTP head"
    if preset.tied is not None:
        tags["embeddings"] = "tied embeddings" if preset.tied else "untied embeddings"
    return tags


def config_lookup(config: dict, keys: tuple[str, ...]) -> Any:
    """First of ``keys`` found in a preset config, searching one nested level.

    The nested level covers families that take a single config object.
    """
    for key in keys:
        if key in config:
            return config[key]
    for value in config.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    return value[key]
    return None


def dimensions(model: nn.Module, mixers: list[nn.Module], config: dict) -> dict:
    """Layers, hidden size, context and vocabulary of a built preset.

    Read from the constructor arguments, falling back to the built modules
    (for example a family that hard-codes its context length).
    """
    dims = {}
    for name, keys in DIM_KEYS.items():
        value = config_lookup(config, keys)
        dims[name] = value if isinstance(value, int) else None
    if dims["layers"] is None and mixers:
        dims["layers"] = len(mixers)
    if dims["context"] is None:
        lengths = [
            m.max_seq_len
            for m in model.modules()
            if isinstance(getattr(m, "max_seq_len", None), int)
        ]
        dims["context"] = max(lengths) if lengths else None
    return dims


def find_guides() -> dict[str, str]:
    """Map family base-class names to the architecture guide that covers them."""
    guides: dict[str, str] = {}
    for path in sorted(MODELS_DOCS.glob("*.md")):
        if path.name == "index.md":
            continue
        for match in GUIDE_MARKER.finditer(path.read_text(encoding="utf-8")):
            for name in match.group(1).split():
                guides[name] = path.name
    return guides


def human(n: int | None) -> str:
    """Format a parameter count: 124.4M, 7.50B, 403.8B, 1.03T."""
    if n is None:
        return "—"
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            value = n / scale
            return f"{value:.2f}{unit}" if value < 10 else f"{value:.1f}{unit}"
    return str(n)


def unique(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def family_tags(family: Family) -> list[str]:
    """Merge the presets' tags per category: ``MHA`` and ``GQA`` read ``MHA / GQA``."""
    tags = []
    for category in TAG_CATEGORIES:
        values = unique(
            [p.tags[category] for p in family.presets if category in p.tags]
        )
        if values:
            tags.append(" / ".join(values))
    return tags


def to_json(families: list[Family]) -> dict:
    presets = [p for f in families for p in f.presets]
    return {
        "generated_by": "scripts/gen_model_zoo.py",
        "note": (
            "Presets are architectures with randomly initialised weights; "
            "OLM does not load pretrained checkpoints."
        ),
        "summary": {
            "families": len(families),
            "presets": len(presets),
            "organisations": len({organisation(f.cls) for f in families}),
        },
        "families": [
            {
                "name": family_name(f.cls),
                "class": f.cls.__name__,
                "module": f.cls.__module__,
                "organisation": organisation(f.cls),
                "source": source_path(f.cls),
                "line": source_line(f.cls),
                "exported": f.exported,
                "guide": f"docs/models/{f.guide}" if f.guide else None,
                "structure": docstring_section(f.cls, "Structure"),
                "tags": family_tags(f),
                "presets": [
                    {
                        "name": p.cls.__name__,
                        "summary": (inspect.getdoc(p.cls) or "").split("\n")[0],
                        "line": source_line(p.cls),
                        "exported": p.exported,
                        "params": {
                            "total": p.params_total,
                            "active": p.params_active,
                            "mtp": p.params_mtp,
                        },
                        "tied_embeddings": p.tied,
                        "token_mixers": [
                            {"layer": label, "count": n} for label, n in p.mixers
                        ],
                        "feed_forward": [
                            {"layer": label, "count": n} for label, n in p.ffns
                        ],
                        "tags": [p.tags[c] for c in TAG_CATEGORIES if c in p.tags],
                        "dims": p.dims,
                        "config": p.config,
                    }
                    for p in f.presets
                ],
            }
            for f in families
        ],
    }


def source_link(cls: type, text: str) -> str:
    return f"[{text}]({BLOB_URL}/{source_path(cls)}#L{source_line(cls)})"


def mix(items: list[tuple[str, int]]) -> str:
    return " + ".join(f"{n} × {label}" for label, n in items) or "—"


def dim(preset: Preset, name: str) -> str:
    value = preset.dims.get(name)
    return "—" if value is None else f"{value:,}"


def to_markdown(families: list[Family]) -> str:
    presets = [p for f in families for p in f.presets]
    organisations = len({organisation(f.cls) for f in families})
    # One link per guide, naming every family it covers.
    guides: dict[str, list[str]] = {}
    for f in families:
        if f.guide:
            guides.setdefault(f.guide, []).append(family_name(f.cls))
    lines = [
        "# Model Zoo",
        "",
        f"OLM implements {len(families)} model families from {organisations} "
        f"organisations as readable PyTorch, with {len(presets)} named presets that "
        "follow the published architecture of a released model. Every number on this "
        "page is generated from the code: each preset is built on PyTorch's `meta` "
        "device and its parameters are counted, so the tables describe what OLM "
        "actually builds.",
        "",
        "<!-- Generated by scripts/gen_model_zoo.py from the code in src/olm/models."
        f" Do not edit by hand: run `{REGENERATE}`. -->",
        "",
        "> [!IMPORTANT]",
        "> **Presets are architectures, not checkpoints.** A preset such as "
        "`Llama3_1_8B()` builds the released model's architecture with **randomly "
        "initialised weights**. OLM does not download or load pretrained weights. "
        "Use a preset to train from scratch, to study or ablate the architecture, or "
        "to check shapes and parameter counts on the `meta` device.",
        "",
        "```python",
        "import torch",
        "from olm.models import Llama3_2_1B",
        "",
        'with torch.device("meta"):  # shapes only: no memory, no weights',
        "    model = Llama3_2_1B()",
        'print(f"{sum(p.numel() for p in model.parameters()):,}")',
        "```",
        "",
    ]
    if guides:
        lines += [
            "Annotated architecture guides walk through the source of "
            + ", ".join(
                f"[{' and '.join(names)}]({guide})" for guide, names in guides.items()
            )
            + ": what changed from the previous generation, the code section by "
            "section, and a tiny model you can train on a CPU.",
            "",
        ]
    lines += [
        "## Families",
        "",
        "One row per configurable base class. *Architecture* is read from the modules "
        "each preset builds.",
        "",
        '<div class="table-scroll">',
        "",
        "| Family | Organisation | Architecture | Presets (total parameters) |",
        "|---|---|---|---|",
    ]
    for f in families:
        name = source_link(f.cls, family_name(f.cls))
        if f.guide:
            name += f" · [guide]({f.guide})"
        preset_cells = " · ".join(
            f"`{p.cls.__name__}` ({human(p.params_total)})" for p in f.presets
        )
        lines.append(
            f"| {name}<br>`{f.cls.__name__}` | {organisation(f.cls)} | "
            f"{' · '.join(family_tags(f))} | {preset_cells or '—'} |"
        )

    lines += [
        "",
        "</div>",
        "",
        "## Preset sizes",
        "",
        "*Total* counts every parameter once (a tied output head shares the embedding "
        "matrix). *Active* is reported for mixture-of-experts models: the parameters "
        "one token passes through, that is the total minus the routed experts it is "
        "not sent to and minus any multi-token-prediction head.",
        "",
        '<div class="table-scroll">',
        "",
        "| Preset | Family | Total | Active | Layers | Hidden | Context | Vocab "
        "| Embeddings |",
        "|---|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    for f in families:
        for p in f.presets:
            total = human(p.params_total)
            if p.params_mtp:
                total += " †"
            tied = "—" if p.tied is None else ("tied" if p.tied else "untied")
            lines.append(
                f"| {source_link(p.cls, p.cls.__name__)} | {family_name(f.cls)} "
                f"| {total} | {human(p.params_active)} "
                f"| {dim(p, 'layers')} | {dim(p, 'hidden')} | {dim(p, 'context')} "
                f"| {dim(p, 'vocab')} | {tied} |"
            )
    lines += [
        "",
        "</div>",
        "",
        "† Includes the multi-token-prediction (MTP) head, an auxiliary training "
        "module that the default `forward` does not use.",
        "",
        "## Layer mix",
        "",
        "The token mixers (attention and its linear-time alternatives) and "
        "feed-forward layers each preset builds, counted per layer. Attention entries "
        "read `query heads/key-value heads`; MLA and the linear-time mixers give "
        "their head count alone (`MLA 128h`); sliding-window entries give the window "
        "in tokens; `MoE 256e top-8 +1 shared` is 256 routed experts of which each "
        "token uses 8, plus one expert every token uses.",
        "",
        '<div class="table-scroll">',
        "",
        "| Preset | Token mixers | Feed-forward |",
        "|---|---|---|",
    ]
    for f in families:
        for p in f.presets:
            lines.append(f"| `{p.cls.__name__}` | {mix(p.mixers)} | {mix(p.ffns)} |")
    lines += [
        "",
        "</div>",
        "",
        "## How this page is made",
        "",
        f"[`scripts/gen_model_zoo.py`]({BLOB_URL}/scripts/gen_model_zoo.py) imports "
        "`olm.models`, treats every class whose `__init__` takes no arguments as a "
        "preset of the base class it inherits from, and builds each preset under "
        '`torch.device("meta")`. It writes this page and '
        f"[`zoo.json`]({BLOB_URL}/docs/models/zoo.json), which also holds every "
        "preset's resolved constructor arguments and the `Structure:` section of "
        "each family's docstring. The website's model pages read the same JSON.",
        "",
        "Parameter counts describe OLM's implementation. Where OLM builds a released "
        "architecture differently (for example, a preset that ties the output head "
        "to the embedding when the release does not), the count follows the code. "
        "Each architecture guide ends with the differences it found for its family.",
        "",
        "After adding or changing a model family, regenerate the page from the "
        "repository root:",
        "",
        "```bash",
        REGENERATE,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit with status 1 if the generated files are stale.",
    )
    args = parser.parse_args()

    families, errors = discover()
    guides = find_guides()
    for family in families:
        family.guide = guides.get(family.cls.__name__)
        for preset in family.presets:
            measure(preset, family.cls)

    outputs = {
        JSON_OUT: json.dumps(to_json(families), indent=2, ensure_ascii=False) + "\n",
        MARKDOWN_OUT: to_markdown(families),
    }

    if errors:
        print("Skipped modules that could not be imported:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)

    if args.check:
        stale = [
            path
            for path, text in outputs.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        if stale:
            print(f"Run: {REGENERATE}", file=sys.stderr)
        return 1 if stale or errors else 0

    MODELS_DOCS.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
    presets = sum(len(f.presets) for f in families)
    print(f"Wrote {len(families)} families and {presets} presets to {MODELS_DOCS}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
