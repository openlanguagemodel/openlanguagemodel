"""CLI for the numerical-parity experiment matrix.

Usage:
    python -m benchmarks.demo2026.parity.run_parity --family all --device cpu \
        --output benchmarks/demo2026/results/raw/parity

One JSON record per (family, seed) is emitted, schema
``benchmarks/demo2026/schemas/parity_result.schema.json``. Status semantics:

    complete    -- run finished; errors are reported as measured
    discrepancy -- run finished but a metric crossed the regression threshold
                   recorded in the config (only used once thresholds exist)
    error       -- run crashed or produced non-finite values

Tolerances are never chosen in advance: raw metrics are always stored.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict

import torch
import yaml

from benchmarks.demo2026 import provenance
from benchmarks.demo2026.parity import compare, models

FAMILIES = ["gpt2", "llama3", "qwen2"]
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "parity")


def load_config(family: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, f"{family}.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def count_parameters(model) -> int:
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total


def run_family(
    family: str,
    device: str,
    seeds: list,
    output_dir: str,
    env: Dict[str, Any],
) -> list:
    config = load_config(family)
    if seeds is None:
        seeds = config["seeds"]

    records = []
    for seed in seeds:
        record: Dict[str, Any] = {
            "family": family,
            "olm_commit": env.get("olm_commit"),
            "reference_library_version": env.get("transformers_version") or "unknown",
            "config": config["model"],
            "input_seed": seed,
            "device": device,
            "dtype": "float32",
            "deterministic_algorithms": True,
            "batch_shape": [config["batch"]["batch_size"], config["batch"]["seq_len"]],
            "early_layer_parameter": config["gradient_probes"]["early"],
            "late_layer_parameter": config["gradient_probes"]["late"],
            "environment": env,
        }
        try:
            models.set_determinism(seed)
            olm_model, hf_model, weight_map, _ = models.build_pair(
                family, config, device=device, init_seed=seed
            )
            record["parameter_count_olm"] = count_parameters(olm_model)
            record["parameter_count_reference"] = count_parameters(hf_model)

            tokens = compare.make_batch(
                seed=seed,
                batch_size=config["batch"]["batch_size"],
                seq_len=config["batch"]["seq_len"],
                vocab_size=config["model"]["vocab_size"],
                device=device,
            )
            metrics = compare.compare_pair(
                olm_model, hf_model, weight_map, tokens, config["gradient_probes"]
            )
            status_hint = metrics.pop("status_hint", None)
            record.update(metrics)
            record["status"] = status_hint or "complete"
        except Exception as exc:  # record failures instead of aborting the matrix
            record.setdefault("parameter_count_olm", 0)
            record.setdefault("parameter_count_reference", 0)
            record.update(
                {
                    "max_logit_absolute_error": math.nan,
                    "mean_logit_absolute_error": math.nan,
                    "loss_absolute_error": math.nan,
                    "embedding_gradient_cosine": math.nan,
                    "early_layer_gradient_cosine": math.nan,
                    "late_layer_gradient_cosine": math.nan,
                    "status": "error",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )

        out_path = os.path.join(output_dir, f"{family}_seed{seed}.json")
        provenance.write_json(out_path, record)
        records.append(record)

        print(
            f"[{family} seed={seed}] status={record['status']} "
            f"max|dlogit|={record['max_logit_absolute_error']:.3e} "
            f"mean|dlogit|={record['mean_logit_absolute_error']:.3e} "
            f"|dloss|={record['loss_absolute_error']:.3e} "
            f"cos(emb/early/late)="
            f"{record['embedding_gradient_cosine']:.10f}/"
            f"{record['early_layer_gradient_cosine']:.10f}/"
            f"{record['late_layer_gradient_cosine']:.10f}"
        )
        if record["status"] == "error":
            print(f"    note: {record.get('notes', '')}", file=sys.stderr)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="all", choices=FAMILIES + ["all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--output", default="benchmarks/demo2026/results/raw/parity"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    env = provenance.capture_environment()
    if env["git_dirty"] and not args.allow_dirty:
        print(
            "WARNING: git worktree is dirty; results are not reportable. "
            "Pass --allow-dirty to acknowledge (debug runs only).",
            file=sys.stderr,
        )
        return 2

    families = FAMILIES if args.family == "all" else [args.family]
    all_records = []
    for family in families:
        all_records.extend(
            run_family(family, args.device, args.seeds, args.output, env)
        )

    n_bad = sum(1 for r in all_records if r["status"] != "complete")
    print(f"\n{len(all_records)} runs, {n_bad} not complete.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
