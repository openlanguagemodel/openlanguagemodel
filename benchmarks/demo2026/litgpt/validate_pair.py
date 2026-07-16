"""Refuse unfair OLM vs LitGPT comparisons."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

import yaml

from benchmarks.demo2026.scaling.run_olm import count_unique_params, load_config
from olm.models.meta.llama3 import Llama3Model


FAIRNESS_KEYS: List[Tuple[str, str, str]] = [
    ("model.vocab_size", "model.vocab_size", "vocab_size"),
    ("model.num_layers", "model.n_layer", "layers"),
    ("model.embed_dim", "model.n_embd", "hidden_size"),
    ("model.num_heads", "model.n_head", "heads"),
    ("model.num_kv_heads", "model.n_query_groups", "query_groups"),
    ("model.intermediate_size", "model.intermediate_size", "intermediate_size"),
    ("training.sequence_length", "training.sequence_length", "sequence_length"),
    ("training.local_batch_size", "training.local_batch_size", "local_batch"),
    ("training.grad_accum_steps", "training.grad_accum_steps", "grad_accum"),
    ("training.precision", "training.precision", "precision"),
    ("training.fused_optimizer", "training.fused_optimizer", "fused_optimizer"),
    ("training.attention_backend", "training.attention_backend", "attention_backend"),
    ("training.compile", "training.compile", "compile"),
    ("training.seed", "training.seed", "seed"),
]


def _get(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def olm_param_count(olm_cfg: Dict[str, Any]) -> int:
    m = olm_cfg["model"]
    model = Llama3Model(
        vocab_size=m["vocab_size"],
        embed_dim=m["embed_dim"],
        intermediate_size=m["intermediate_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        num_kv_heads=m["num_kv_heads"],
        max_seq_len=m["max_seq_len"],
        rope_theta=m["rope_theta"],
        dropout=m.get("dropout", 0.0),
        tie_weights=m.get("tie_weights", True),
    )
    return count_unique_params(model)


def validate(olm_cfg: Dict[str, Any], lit_cfg: Dict[str, Any]) -> List[str]:
    errors = []
    for olm_path, lit_path, label in FAIRNESS_KEYS:
        ov = _get(olm_cfg, olm_path)
        lv = _get(lit_cfg, lit_path)
        if ov != lv:
            errors.append(f"{label}: olm={ov!r} litgpt={lv!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olm-config", required=True)
    parser.add_argument("--litgpt-config", required=True)
    args = parser.parse_args()

    olm_cfg = load_config(args.olm_config)
    with open(args.litgpt_config) as fh:
        lit_cfg = yaml.safe_load(fh)

    errors = validate(olm_cfg, lit_cfg)
    olm_n = olm_param_count(olm_cfg)

    if errors:
        print("FAIRNESS CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Fairness check passed for static config fields.")
    print(f"  olm_unique_parameters={olm_n} (verify LitGPT count at runtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
