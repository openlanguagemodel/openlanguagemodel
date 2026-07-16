"""CLI for the breadth validation experiment.

Validates all 9 families and 27 named presets of OpenLanguageModel without
allocating the actual large models. Two kinds of checks are performed:

**Preset checks** (no real allocation of large models):
  constructor_config_matches_manifest  -- patch base __init__, call preset(),
                                          compare captured kwargs to manifest
  formula_param_count                  -- formula evaluated on preset's kwargs
  formula_param_count_in_expected_range-- formula count within published-size range
  tie_setting_matches_manifest         -- tie_weights default is True

**Family checks** (tiny reduced model, runs on CPU):
  reduced_forward_smoke                -- build tiny model, verify forward shape
  reduced_param_count_matches_formula  -- actual numel() == formula(reduced_config)
  tied_embedding_behavior              -- lm_head weight IS the embedding weight
  checkpoint_roundtrip_bitwise         -- save + load + logits identical bitwise

Usage::

    python -m benchmarks.demo2026.breadth.run_breadth \\
        --output benchmarks/demo2026/results/raw/breadth_result.json

    python -m benchmarks.demo2026.breadth.run_breadth --allow-dirty
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch

from benchmarks.demo2026 import provenance
from benchmarks.demo2026.breadth.manifest import ALL_FAMILIES, FamilySpec, PresetSpec
from benchmarks.demo2026.breadth.param_formulas import (
    FORMULA_REGISTRY,
    count_params_unique,
)

_PASS = "pass"
_FAIL = "fail"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pf(ok: bool) -> str:
    return _PASS if ok else _FAIL


def _count_failed(checks: Dict[str, Any]) -> int:
    return sum(1 for v in checks.values() if v == _FAIL)


def _get_embedding_weight(model, family_name: str) -> torch.nn.Parameter:
    """Return the token-embedding weight parameter for a model instance."""
    if family_name == "gpt2":
        # GPT2Model: blocks[0] = Block([token_embedding, AbsPos])
        return model.blocks[0].blocks[0].embedding.weight
    elif family_name == "opt":
        # OPTModel: blocks[0] = token_embedding Embedding module
        return model.blocks[0].embedding.weight
    else:
        # Most models: blocks[0] = Embedding
        return model.blocks[0].embedding.weight


def _get_lm_head_weight(model, family_name: str) -> torch.nn.Parameter:
    """Return the lm-head projection weight parameter for a model instance."""
    if family_name == "gpt2":
        # GPT2Model: blocks[2] = OutputHead
        return model.blocks[2].weight
    elif family_name == "opt":
        # OPTModel: blocks[5] = OutputHead  (tok_emb, pos_emb, dropout, Repeat, LN, head)
        return model.blocks[5].weight
    elif family_name == "gemma2":
        # Gemma2Model: blocks[3] = OutputHead  (embed, Repeat, RMSNorm, head, softcap)
        return model.blocks[3].weight
    else:
        # Llama2/3, Qwen2, Phi3/4, OLMo: blocks[3] = OutputHead
        return model.blocks[3].weight


# ---------------------------------------------------------------------------
# Preset checks (no large-model allocation)
# ---------------------------------------------------------------------------


def check_preset(
    family: FamilySpec,
    preset: PresetSpec,
    formula_fn,
) -> Dict[str, Any]:
    """Validate one preset via constructor patching and formula evaluation."""
    checks: Dict[str, Any] = {}
    notes: List[str] = []

    # 1. constructor_config_matches_manifest
    try:
        mod = importlib.import_module(family.module_path)
        base_cls = getattr(mod, family.base_class)
        preset_cls = getattr(mod, preset.name)

        with patch.object(base_cls, "__init__", return_value=None) as mock_init:
            preset_cls()

        if mock_init.call_count != 1:
            checks["constructor_config_matches_manifest"] = _FAIL
            notes.append(f"base __init__ called {mock_init.call_count} times, expected 1")
        else:
            captured = mock_init.call_args.kwargs
            mismatches = {}
            for key, expected_val in preset.expected_kwargs.items():
                if key not in captured:
                    mismatches[key] = f"missing (expected {expected_val!r})"
                elif captured[key] != expected_val:
                    mismatches[key] = f"got {captured[key]!r}, expected {expected_val!r}"
            extra_keys = {
                k for k in captured
                if k not in preset.expected_kwargs and k not in ("tie_weights", "dropout")
            }
            if extra_keys:
                mismatches["_extra_keys"] = sorted(extra_keys)
            checks["constructor_config_matches_manifest"] = _pf(not mismatches)
            if mismatches:
                notes.append(f"kwarg mismatches: {mismatches}")
    except Exception as exc:
        checks["constructor_config_matches_manifest"] = _FAIL
        notes.append(f"constructor patch failed: {exc}")

    # 2. formula_param_count + formula_param_count_in_expected_range
    try:
        formula_count = formula_fn(**preset.expected_kwargs)
        checks["formula_param_count"] = formula_count
        in_range = (
            (preset.param_lo == 0 and preset.param_hi == 0)
            or (preset.param_lo <= formula_count <= preset.param_hi)
        )
        checks["formula_param_count_in_expected_range"] = _pf(in_range)
        if not in_range:
            notes.append(
                f"formula count {formula_count:,} outside "
                f"[{preset.param_lo:,}, {preset.param_hi:,}]"
            )
    except Exception as exc:
        checks["formula_param_count"] = None
        checks["formula_param_count_in_expected_range"] = _FAIL
        notes.append(f"formula evaluation failed: {exc}")

    # 3. tie_setting_matches_manifest (default tie_weights=True)
    try:
        mod = importlib.import_module(family.module_path)
        base_cls = getattr(mod, family.base_class)
        preset_cls = getattr(mod, preset.name)

        with patch.object(base_cls, "__init__", return_value=None) as mock_init:
            preset_cls()

        captured = mock_init.call_args.kwargs
        actual_tie = captured.get("tie_weights", True)
        checks["tie_setting_matches_manifest"] = _pf(actual_tie == preset.tie_weights)
        if actual_tie != preset.tie_weights:
            notes.append(
                f"tie_weights: got {actual_tie!r}, expected {preset.tie_weights!r}"
            )
    except Exception as exc:
        checks["tie_setting_matches_manifest"] = _FAIL
        notes.append(f"tie_weights check failed: {exc}")

    return {
        "family": family.name,
        "preset": preset.name,
        "checks": checks,
        **({"notes": "; ".join(notes)} if notes else {}),
    }


# ---------------------------------------------------------------------------
# Family checks (reduced model, CPU)
# ---------------------------------------------------------------------------


def _build_reduced(family: FamilySpec):
    """Instantiate the family base class with reduced_config."""
    mod = importlib.import_module(family.module_path)
    base_cls = getattr(mod, family.base_class)
    return base_cls(**family.reduced_config)


def check_family(family: FamilySpec, formula_fn, tmp_dir: str) -> Dict[str, Any]:
    """Run all four family-level checks on a tiny reduced model."""
    checks: Dict[str, Any] = {}
    notes: List[str] = []
    cfg = family.reduced_config

    # 1. reduced_forward_smoke
    try:
        model = _build_reduced(family)
        model.eval()
        vocab = cfg["vocab_size"]
        seq = cfg.get("max_seq_len", 16)
        seq = min(seq, 16)
        tokens = torch.randint(0, vocab, (1, seq))
        with torch.no_grad():
            logits = model(tokens)
        expected_shape = (1, seq, vocab)
        ok = logits.shape == expected_shape
        checks["reduced_forward_smoke"] = _pf(ok)
        if not ok:
            notes.append(f"logits shape {tuple(logits.shape)} != {expected_shape}")
    except Exception as exc:
        checks["reduced_forward_smoke"] = _FAIL
        notes.append(f"forward smoke failed: {traceback.format_exc(limit=3)}")
        # Skip subsequent checks that require a working model
        checks["reduced_param_count_matches_formula"] = _FAIL
        checks["tied_embedding_behavior"] = _FAIL
        checks["checkpoint_roundtrip_bitwise"] = _FAIL
        return {
            "family": family.name,
            "checks": checks,
            "reduced_config": cfg,
            "notes": "; ".join(notes),
        }

    # 2. reduced_param_count_matches_formula
    try:
        actual = count_params_unique(model)
        expected = formula_fn(**cfg)
        ok = actual == expected
        checks["reduced_param_count_matches_formula"] = _pf(ok)
        if not ok:
            notes.append(
                f"actual params {actual:,} != formula {expected:,} "
                f"(delta {actual - expected:+,})"
            )
    except Exception as exc:
        checks["reduced_param_count_matches_formula"] = _FAIL
        notes.append(f"param count check failed: {exc}")

    # 3. tied_embedding_behavior
    try:
        emb_w = _get_embedding_weight(model, family.name)
        head_w = _get_lm_head_weight(model, family.name)
        ok = emb_w is head_w
        checks["tied_embedding_behavior"] = _pf(ok)
        if not ok:
            notes.append("embedding weight is NOT the same object as lm_head weight")
    except Exception as exc:
        checks["tied_embedding_behavior"] = _FAIL
        notes.append(f"tying check failed: {exc}")

    # 4. checkpoint_roundtrip_bitwise
    try:
        save_dir = os.path.join(tmp_dir, f"{family.name}_ckpt")
        model.eval()
        vocab = cfg["vocab_size"]
        seq_len = min(cfg.get("max_seq_len", 16), 16)
        tokens = torch.randint(0, vocab, (1, seq_len))

        with torch.no_grad():
            logits_before = model(tokens).clone()

        model.save(save_dir)

        from olm.nn.structure.block import load_block

        loaded = load_block(save_dir, trusted=True)
        loaded.eval()

        with torch.no_grad():
            logits_after = loaded(tokens)

        ok = torch.equal(logits_before, logits_after)
        checks["checkpoint_roundtrip_bitwise"] = _pf(ok)
        if not ok:
            max_diff = (logits_before - logits_after).abs().max().item()
            notes.append(f"logits differ after round-trip; max |diff| = {max_diff:.3e}")
    except Exception as exc:
        checks["checkpoint_roundtrip_bitwise"] = _FAIL
        notes.append(f"checkpoint roundtrip failed: {traceback.format_exc(limit=3)}")

    return {
        "family": family.name,
        "checks": checks,
        "reduced_config": cfg,
        **({"notes": "; ".join(notes)} if notes else {}),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all(env: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    family_records: List[Dict[str, Any]] = []
    preset_records: List[Dict[str, Any]] = []

    for family in ALL_FAMILIES:
        formula_fn = FORMULA_REGISTRY[family.formula]

        print(f"\n[family:{family.name}] running family checks …", flush=True)
        try:
            frec = check_family(family, formula_fn, tmp_dir)
        except Exception as exc:
            frec = {
                "family": family.name,
                "checks": {
                    "reduced_forward_smoke": _FAIL,
                    "reduced_param_count_matches_formula": _FAIL,
                    "tied_embedding_behavior": _FAIL,
                    "checkpoint_roundtrip_bitwise": _FAIL,
                },
                "reduced_config": family.reduced_config,
                "notes": f"unexpected error: {exc}",
            }
        family_records.append(frec)
        n_fail = _count_failed(frec["checks"])
        status = "ok" if n_fail == 0 else f"{n_fail} FAILED"
        print(f"  [{family.name}] family → {status}", flush=True)
        if frec.get("notes"):
            print(f"    note: {frec['notes']}", file=sys.stderr)

        for preset in family.presets:
            prec = check_preset(family, preset, formula_fn)
            preset_records.append(prec)
            n_pfail = _count_failed(
                {k: v for k, v in prec["checks"].items() if isinstance(v, str)}
            )
            pstatus = "ok" if n_pfail == 0 else f"{n_pfail} FAILED"
            fpc = prec["checks"].get("formula_param_count")
            fpc_str = f"{fpc:,}" if fpc is not None else "n/a"
            print(
                f"  [{family.name}] preset={preset.name} → {pstatus}  formula={fpc_str}",
                flush=True,
            )
            if prec.get("notes"):
                print(f"    note: {prec['notes']}", file=sys.stderr)

    all_checks = (
        [v for r in family_records for v in r["checks"].values() if isinstance(v, str)]
        + [
            v
            for r in preset_records
            for v in r["checks"].values()
            if isinstance(v, str)
        ]
    )
    n_failed = sum(1 for v in all_checks if v == _FAIL)

    summary = {
        "n_families": len(family_records),
        "n_presets": len(preset_records),
        "n_failed_checks": n_failed,
    }

    return {
        "olm_commit": env.get("olm_commit"),
        "environment": env,
        "families": family_records,
        "presets": preset_records,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/demo2026/results/raw/breadth_result.json",
        help="Path for the JSON result file.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run even if the git worktree is dirty (non-reportable).",
    )
    args = parser.parse_args()

    env = provenance.capture_environment()

    if env["git_dirty"] and not args.allow_dirty:
        print(
            "WARNING: git worktree is dirty; results are not reportable. "
            "Pass --allow-dirty to acknowledge.",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="olm_breadth_") as tmp_dir:
        result = run_all(env, tmp_dir)

    provenance.write_json(args.output, result)
    n_fail = result["summary"]["n_failed_checks"]
    n_total = len(result["families"]) * 4 + len(result["presets"]) * 3
    print(
        f"\n[breadth] {len(result['families'])} families, "
        f"{len(result['presets'])} presets, "
        f"{n_fail}/{n_total} checks failed.",
        flush=True,
    )
    print(f"[breadth] result written to {args.output}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
