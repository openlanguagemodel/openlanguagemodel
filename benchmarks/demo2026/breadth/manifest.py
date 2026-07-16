"""Declarative manifest for all 9 families and 27 named presets.

PresetSpec captures:
  - expected_kwargs: kwargs passed to the base-class __init__ (for constructor check)
  - tie_weights: expected weight-tying default (True for all 27 documented presets)
  - param_lo / param_hi: loose published-size range for formula sanity check

FamilySpec captures:
  - reduced_config: tiny kwargs to build a smoke-testable model on CPU
  - formula: name of the function in param_formulas.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class PresetSpec:
    name: str
    expected_kwargs: Dict[str, Any]
    tie_weights: bool = True
    param_lo: int = 0
    param_hi: int = 0


@dataclass
class FamilySpec:
    name: str
    display_name: str
    module_path: str
    base_class: str
    formula: str
    reduced_config: Dict[str, Any]
    presets: List[PresetSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPT-2 family
# ---------------------------------------------------------------------------
_GPT2 = FamilySpec(
    name="gpt2",
    display_name="GPT-2",
    module_path="olm.models.openai.gpt2",
    base_class="GPT2Model",
    formula="gpt2_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=16,
        dropout=0.0,
    ),
    presets=[
        PresetSpec(
            "GPT2",
            dict(vocab_size=50257, embed_dim=768, num_layers=12, num_heads=12, max_seq_len=1024),
            param_lo=62_000_000,
            param_hi=248_000_000,
        ),
        PresetSpec(
            "GPT2Medium",
            dict(vocab_size=50257, embed_dim=1024, num_layers=24, num_heads=16, max_seq_len=1024),
            param_lo=177_000_000,
            param_hi=710_000_000,
        ),
        PresetSpec(
            "GPT2Large",
            dict(vocab_size=50257, embed_dim=1280, num_layers=36, num_heads=20, max_seq_len=1024),
            param_lo=387_000_000,
            param_hi=1_548_000_000,
        ),
        PresetSpec(
            "GPT2XL",
            dict(vocab_size=50257, embed_dim=1600, num_layers=48, num_heads=25, max_seq_len=1024),
            param_lo=779_000_000,
            param_hi=3_116_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Llama 2 family
# ---------------------------------------------------------------------------
_LLAMA2 = FamilySpec(
    name="llama2",
    display_name="Llama 2",
    module_path="olm.models.meta.llama2",
    base_class="Llama2Model",
    formula="llama_swiglu_gqa_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "Llama2_7B",
            dict(
                vocab_size=32000, embed_dim=4096, intermediate_size=11008,
                num_layers=32, num_heads=32, num_kv_heads=32,
                max_seq_len=4096, rope_theta=10000.0,
            ),
            param_lo=3_500_000_000,
            param_hi=14_000_000_000,
        ),
        PresetSpec(
            "Llama2_13B",
            dict(
                vocab_size=32000, embed_dim=5120, intermediate_size=13824,
                num_layers=40, num_heads=40, num_kv_heads=40,
                max_seq_len=4096, rope_theta=10000.0,
            ),
            param_lo=6_500_000_000,
            param_hi=26_000_000_000,
        ),
        PresetSpec(
            "Llama2_70B",
            dict(
                vocab_size=32000, embed_dim=8192, intermediate_size=28672,
                num_layers=80, num_heads=64, num_kv_heads=8,
                max_seq_len=4096, rope_theta=10000.0,
            ),
            param_lo=35_000_000_000,
            param_hi=140_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Llama 3 family (Llama 3.1 and 3.2)
# ---------------------------------------------------------------------------
_LLAMA3 = FamilySpec(
    name="llama3",
    display_name="Llama 3",
    module_path="olm.models.meta.llama3",
    base_class="Llama3Model",
    formula="llama_swiglu_gqa_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "Llama3_1_8B",
            dict(
                vocab_size=128256, embed_dim=4096, intermediate_size=14336,
                num_layers=32, num_heads=32, num_kv_heads=8,
                max_seq_len=131072, rope_theta=500000.0,
            ),
            param_lo=4_000_000_000,
            param_hi=16_000_000_000,
        ),
        PresetSpec(
            "Llama3_1_70B",
            dict(
                vocab_size=128256, embed_dim=8192, intermediate_size=28672,
                num_layers=80, num_heads=64, num_kv_heads=8,
                max_seq_len=131072, rope_theta=500000.0,
            ),
            param_lo=35_000_000_000,
            param_hi=140_000_000_000,
        ),
        PresetSpec(
            "Llama3_1_405B",
            dict(
                vocab_size=128256, embed_dim=16384, intermediate_size=53248,
                num_layers=126, num_heads=128, num_kv_heads=8,
                max_seq_len=131072, rope_theta=500000.0,
            ),
            param_lo=200_000_000_000,
            param_hi=810_000_000_000,
        ),
        PresetSpec(
            "Llama3_2_1B",
            dict(
                vocab_size=128256, embed_dim=2048, intermediate_size=8192,
                num_layers=16, num_heads=32, num_kv_heads=8,
                max_seq_len=131072, rope_theta=500000.0,
            ),
            param_lo=500_000_000,
            param_hi=2_000_000_000,
        ),
        PresetSpec(
            "Llama3_2_3B",
            dict(
                vocab_size=128256, embed_dim=3072, intermediate_size=8192,
                num_layers=28, num_heads=24, num_kv_heads=8,
                max_seq_len=131072, rope_theta=500000.0,
            ),
            param_lo=1_500_000_000,
            param_hi=6_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Qwen 2.5 family
# ---------------------------------------------------------------------------
_QWEN2 = FamilySpec(
    name="qwen2",
    display_name="Qwen 2.5",
    module_path="olm.models.alibaba.qwen2",
    base_class="Qwen2Model",
    formula="qwen2_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        rope_theta=10000.0,
    ),
    presets=[
        PresetSpec(
            "Qwen2_5_0_5B",
            dict(
                vocab_size=151936, embed_dim=896, intermediate_size=4864,
                num_layers=24, num_heads=14, num_kv_heads=2,
                max_seq_len=32768, rope_theta=1000000.0,
            ),
            param_lo=250_000_000,
            param_hi=1_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_1_5B",
            dict(
                vocab_size=151936, embed_dim=1536, intermediate_size=8960,
                num_layers=28, num_heads=12, num_kv_heads=2,
                max_seq_len=131072, rope_theta=1000000.0,
            ),
            param_lo=750_000_000,
            param_hi=3_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_3B",
            dict(
                vocab_size=151936, embed_dim=2048, intermediate_size=11008,
                num_layers=36, num_heads=16, num_kv_heads=2,
                max_seq_len=32768, rope_theta=1000000.0,
            ),
            param_lo=1_500_000_000,
            param_hi=6_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_7B",
            dict(
                vocab_size=152064, embed_dim=3584, intermediate_size=18944,
                num_layers=28, num_heads=28, num_kv_heads=4,
                max_seq_len=131072, rope_theta=1000000.0,
            ),
            param_lo=3_500_000_000,
            param_hi=14_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_14B",
            dict(
                vocab_size=152064, embed_dim=5120, intermediate_size=13824,
                num_layers=48, num_heads=40, num_kv_heads=8,
                max_seq_len=131072, rope_theta=1000000.0, rms_norm_eps=1e-5,
            ),
            param_lo=7_000_000_000,
            param_hi=28_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_32B",
            dict(
                vocab_size=152064, embed_dim=5120, intermediate_size=27648,
                num_layers=64, num_heads=40, num_kv_heads=8,
                max_seq_len=131072, rope_theta=1000000.0, rms_norm_eps=1e-5,
            ),
            param_lo=16_000_000_000,
            param_hi=64_000_000_000,
        ),
        PresetSpec(
            "Qwen2_5_72B",
            dict(
                vocab_size=152064, embed_dim=8192, intermediate_size=29568,
                num_layers=80, num_heads=64, num_kv_heads=8,
                max_seq_len=131072, rope_theta=1000000.0, rms_norm_eps=1e-5,
            ),
            param_lo=36_000_000_000,
            param_hi=144_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Phi-3 family
# ---------------------------------------------------------------------------
_PHI3 = FamilySpec(
    name="phi3",
    display_name="Phi-3",
    module_path="olm.models.microsoft.phi3",
    base_class="Phi3Model",
    formula="llama_swiglu_gqa_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "Phi3_5_Mini",
            dict(
                vocab_size=32064, embed_dim=3072, intermediate_size=8192,
                num_layers=32, num_heads=32, num_kv_heads=32,
                max_seq_len=131072, rope_theta=10000.0, activation="swiglu",
            ),
            param_lo=1_900_000_000,
            param_hi=7_600_000_000,
        ),
        PresetSpec(
            "Phi3_Small",
            dict(
                vocab_size=100352, embed_dim=4096, intermediate_size=14336,
                num_layers=32, num_heads=32, num_kv_heads=8,
                max_seq_len=131072, rope_theta=1000000.0, activation="geglu",
            ),
            param_lo=3_500_000_000,
            param_hi=14_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Phi-4 family
# ---------------------------------------------------------------------------
_PHI4 = FamilySpec(
    name="phi4",
    display_name="Phi-4",
    module_path="olm.models.microsoft.phi4",
    base_class="Phi4Model",
    formula="llama_swiglu_gqa_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "Phi4_14B",
            dict(
                vocab_size=100352, embed_dim=5120, intermediate_size=17920,
                num_layers=40, num_heads=40, num_kv_heads=10,
                max_seq_len=16384, rope_theta=250000.0,
            ),
            param_lo=7_000_000_000,
            param_hi=28_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Gemma 2 family
# ---------------------------------------------------------------------------
_GEMMA2 = FamilySpec(
    name="gemma2",
    display_name="Gemma 2",
    module_path="olm.models.google.gemma2",
    base_class="Gemma2Model",
    formula="gemma2_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "Gemma2_2B",
            dict(
                vocab_size=256000, embed_dim=2304, intermediate_size=9216,
                num_layers=26, num_heads=8, num_kv_heads=4,
                head_dim=256, max_seq_len=8192, rope_theta=10000.0,
            ),
            param_lo=1_000_000_000,
            param_hi=4_000_000_000,
        ),
        PresetSpec(
            "Gemma2_9B",
            dict(
                vocab_size=256000, embed_dim=3584, intermediate_size=14336,
                num_layers=42, num_heads=16, num_kv_heads=8,
                head_dim=256, max_seq_len=8192, rope_theta=10000.0,
            ),
            param_lo=4_500_000_000,
            param_hi=18_000_000_000,
        ),
        PresetSpec(
            "Gemma2_27B",
            dict(
                vocab_size=256000, embed_dim=4608, intermediate_size=36864,
                num_layers=46, num_heads=32, num_kv_heads=16,
                head_dim=128, max_seq_len=8192, rope_theta=10000.0,
            ),
            param_lo=13_500_000_000,
            param_hi=54_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# OLMo family
# ---------------------------------------------------------------------------
_OLMO = FamilySpec(
    name="olmo",
    display_name="OLMo",
    module_path="olm.models.allenai.olmo",
    base_class="OLMoModel",
    formula="olmo_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=16,
    ),
    presets=[
        PresetSpec(
            "OLMo_7B",
            dict(
                vocab_size=50280, embed_dim=4096, intermediate_size=22016,
                num_layers=32, num_heads=32, max_seq_len=2048,
            ),
            # OLM uses intermediate=22016 (2× the standard OLMo hidden_dim=11008),
            # giving ~11B formula count; use a generous range around that.
            param_lo=3_500_000_000,
            param_hi=14_000_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# OPT family
# ---------------------------------------------------------------------------
_OPT = FamilySpec(
    name="opt",
    display_name="OPT",
    module_path="olm.models.facebook.opt",
    base_class="OPTModel",
    formula="opt_params",
    reduced_config=dict(
        vocab_size=128,
        embed_dim=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ),
    presets=[
        PresetSpec(
            "OPT125M",
            dict(
                vocab_size=50272, embed_dim=768, intermediate_size=3072,
                num_layers=12, num_heads=12, dropout=0.1,
            ),
            param_lo=62_000_000,
            param_hi=250_000_000,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Master list (order determines JSON output)
# ---------------------------------------------------------------------------
ALL_FAMILIES: List[FamilySpec] = [
    _GPT2,
    _LLAMA2,
    _LLAMA3,
    _QWEN2,
    _PHI3,
    _PHI4,
    _GEMMA2,
    _OLMO,
    _OPT,
]

PRESET_COUNT = sum(len(f.presets) for f in ALL_FAMILIES)
FAMILY_COUNT = len(ALL_FAMILIES)
