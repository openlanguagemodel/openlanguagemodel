"""Parametrized pytest suite for the breadth validation package.

Tests are intentionally safe on a laptop: no large-model allocation.

Coverage:
  - All 27 presets: constructor kwargs match the manifest (patch-based)
  - All 27 presets: tie_weights defaults to True
  - All 27 presets: formula_param_count > 0 and finite
  - All 9 families: tiny reduced forward pass succeeds
  - All 9 families: actual unique param count == formula(reduced_config)
  - All 9 families: embedding weight IS the lm_head weight (tying)
  - One reduced model per family: checkpoint round-trip is bitwise identical
"""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any, Dict, Tuple
from unittest.mock import patch

import pytest
import torch

from benchmarks.demo2026.breadth.manifest import ALL_FAMILIES, FamilySpec, PresetSpec
from benchmarks.demo2026.breadth.param_formulas import (
    FORMULA_REGISTRY,
    count_params_unique,
)
from benchmarks.demo2026.breadth.run_breadth import (
    _build_reduced,
    _get_embedding_weight,
    _get_lm_head_weight,
)


# ---------------------------------------------------------------------------
# Parametrize fixtures
# ---------------------------------------------------------------------------


def _all_presets() -> list:
    """Flat list of (family, preset) pairs for parametrization."""
    return [
        (family, preset)
        for family in ALL_FAMILIES
        for preset in family.presets
    ]


def _preset_ids() -> list:
    return [f"{f.name}/{p.name}" for f, p in _all_presets()]


def _all_families() -> list:
    return list(ALL_FAMILIES)


def _family_ids() -> list:
    return [f.name for f in _all_families()]


# ---------------------------------------------------------------------------
# Preset-level tests  (no large-model allocation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,preset", _all_presets(), ids=_preset_ids())
def test_preset_constructor_kwargs_match_manifest(
    family: FamilySpec, preset: PresetSpec
):
    """Verify that the preset's __init__ passes exactly the expected kwargs."""
    mod = importlib.import_module(family.module_path)
    base_cls = getattr(mod, family.base_class)
    preset_cls = getattr(mod, preset.name)

    with patch.object(base_cls, "__init__", return_value=None) as mock_init:
        preset_cls()

    assert mock_init.call_count == 1, (
        f"{preset.name}: base __init__ called {mock_init.call_count} times"
    )
    captured = mock_init.call_args.kwargs
    for key, expected_val in preset.expected_kwargs.items():
        assert key in captured, f"{preset.name}: missing kwarg '{key}'"
        assert captured[key] == expected_val, (
            f"{preset.name}: kwarg '{key}' is {captured[key]!r}, "
            f"expected {expected_val!r}"
        )


@pytest.mark.parametrize("family,preset", _all_presets(), ids=_preset_ids())
def test_preset_tie_weights_default(family: FamilySpec, preset: PresetSpec):
    """All documented presets use the default tie_weights=True."""
    mod = importlib.import_module(family.module_path)
    base_cls = getattr(mod, family.base_class)
    preset_cls = getattr(mod, preset.name)

    with patch.object(base_cls, "__init__", return_value=None) as mock_init:
        preset_cls()

    captured = mock_init.call_args.kwargs
    assert captured.get("tie_weights", True) is True, (
        f"{preset.name}: tie_weights={captured.get('tie_weights')!r}, expected True"
    )


@pytest.mark.parametrize("family,preset", _all_presets(), ids=_preset_ids())
def test_preset_formula_param_count_positive(
    family: FamilySpec, preset: PresetSpec
):
    """Formula evaluation on preset kwargs yields a positive finite integer."""
    formula_fn = FORMULA_REGISTRY[family.formula]
    count = formula_fn(**preset.expected_kwargs)
    assert isinstance(count, int), f"{preset.name}: formula returned non-int {count!r}"
    assert count > 0, f"{preset.name}: formula returned non-positive {count}"


@pytest.mark.parametrize("family,preset", _all_presets(), ids=_preset_ids())
def test_preset_formula_param_count_in_range(
    family: FamilySpec, preset: PresetSpec
):
    """Formula count falls within the manifest's published-size range."""
    if preset.param_lo == 0 and preset.param_hi == 0:
        pytest.skip("no range defined for this preset")
    formula_fn = FORMULA_REGISTRY[family.formula]
    count = formula_fn(**preset.expected_kwargs)
    assert preset.param_lo <= count <= preset.param_hi, (
        f"{preset.name}: formula count {count:,} outside "
        f"[{preset.param_lo:,}, {preset.param_hi:,}]"
    )


# ---------------------------------------------------------------------------
# Family-level tests (tiny reduced models)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", _all_families(), ids=_family_ids())
def test_family_reduced_forward_smoke(family: FamilySpec):
    """Build tiny model, run forward, check output shape."""
    model = _build_reduced(family)
    model.eval()
    cfg = family.reduced_config
    vocab = cfg["vocab_size"]
    seq_len = min(cfg.get("max_seq_len", 16), 16)
    tokens = torch.randint(0, vocab, (1, seq_len))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (1, seq_len, vocab), (
        f"{family.name}: got {tuple(logits.shape)}, "
        f"expected (1, {seq_len}, {vocab})"
    )


@pytest.mark.parametrize("family", _all_families(), ids=_family_ids())
def test_family_reduced_param_count_matches_formula(family: FamilySpec):
    """De-duplicated param count of reduced model matches closed-form formula."""
    model = _build_reduced(family)
    formula_fn = FORMULA_REGISTRY[family.formula]
    actual = count_params_unique(model)
    expected = formula_fn(**family.reduced_config)
    assert actual == expected, (
        f"{family.name}: actual {actual:,} != formula {expected:,} "
        f"(delta {actual - expected:+,})"
    )


@pytest.mark.parametrize("family", _all_families(), ids=_family_ids())
def test_family_tied_embedding_identity(family: FamilySpec):
    """lm_head weight is literally the same tensor as the token embedding."""
    model = _build_reduced(family)
    emb_w = _get_embedding_weight(model, family.name)
    head_w = _get_lm_head_weight(model, family.name)
    assert emb_w is head_w, (
        f"{family.name}: embedding weight id={id(emb_w)} != "
        f"lm_head weight id={id(head_w)}"
    )


@pytest.mark.parametrize("family", _all_families(), ids=_family_ids())
def test_family_checkpoint_roundtrip_bitwise(family: FamilySpec, tmp_path):
    """Save → load → forward yields bitwise-identical logits on CPU."""
    from olm.nn.structure.block import load_block

    model = _build_reduced(family)
    model.eval()
    cfg = family.reduced_config
    vocab = cfg["vocab_size"]
    seq_len = min(cfg.get("max_seq_len", 16), 16)
    tokens = torch.randint(0, vocab, (1, seq_len))

    with torch.no_grad():
        logits_before = model(tokens).clone()

    save_dir = str(tmp_path / "ckpt")
    model.save(save_dir)

    loaded = load_block(save_dir, trusted=True)
    loaded.eval()

    with torch.no_grad():
        logits_after = loaded(tokens)

    assert torch.equal(logits_before, logits_after), (
        f"{family.name}: logits differ after round-trip; "
        f"max |diff| = {(logits_before - logits_after).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Coverage sanity: manifest counts
# ---------------------------------------------------------------------------


def test_manifest_has_27_presets():
    total = sum(len(f.presets) for f in ALL_FAMILIES)
    assert total == 27, f"Expected 27 presets, got {total}"


def test_manifest_has_9_families():
    assert len(ALL_FAMILIES) == 9, f"Expected 9 families, got {len(ALL_FAMILIES)}"


def test_all_family_formulas_registered():
    for family in ALL_FAMILIES:
        assert family.formula in FORMULA_REGISTRY, (
            f"{family.name}: formula '{family.formula}' not in FORMULA_REGISTRY"
        )
