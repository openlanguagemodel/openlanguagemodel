#!/usr/bin/env python3
"""Summarize the non-aggregated benchmark catalog into per-cell results.

A cell is one (GPU, model, context length) point. The catalog keeps every
measured batch for every seed, including out-of-memory (OOM) attempts. Tables
and the website need one answer per cell, chosen by a fixed rule:

1. keep complete, warmup-stabilized rows and group them by batch size;
2. take the median of each metric across the seeds that completed that batch;
3. consider only batches that every seed with rows in the cell completed, so
   a batch that fit one seed at the edge of memory is never the headline;
4. select the batch with the highest median tokens/second (larger batch on a
   tie). For a fixed cell, MFU is proportional to tokens/second, so the same
   batch also has the highest MFU.

A cell with rows but no such batch is reported as OOM when an attempt ran out
of memory, and as incomplete otherwise. Cells without rows were not run: the
context sweep only extends models that trained at 2,048 tokens, and
16,384/32,768 tokens only for models that fit the previous length.

The 2,048-token sweep doubles the batch from 1 until it runs out of memory,
two doublings in a row fail to add 1% throughput, or it reaches 64. Other
contexts test only three batches around the token-equivalent batch. When the
largest tested batch fits every seed and is also the fastest, a larger batch
was simply not tried; such cells are flagged ``search_capped`` so consumers
can say so.

Prices are joined here as well so every consumer derives cost the same way.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_ROOT / "data"
CATALOG_PATH = DATA_DIR / "benchmark_catalog.csv"
RUNPOD_PRICING_PATH = DATA_DIR / "runpod_pricing_snapshot.json"
PROVIDER_PRICING_PATH = DATA_DIR / "provider_pricing_snapshot.json"
SCALING_PATH = DATA_DIR / "multi_gpu_scaling.json"

MODEL_ORDER = ["150m", "250m", "350m", "500m", "700m", "1b", "2b", "4b", "8b"]
PROTOCOL_SEEDS = (11, 22)
OOM_STATUSES = {"oom", "oom_during_model_build"}

# GPUs that are wanted in the catalog but have no rows yet. They are shown as
# "not yet measured" instead of being silently absent, and drop off this list
# on their own once rows for them are merged.
NOT_YET_MEASURED = {
    "Tesla T4": {
        "display_name": "T4",
        "vram_gb": 16,
        "amp_dtype": "fp16",
        "note": (
            "Free Colab GPU. It has no BF16 Tensor Cores, so it runs the harness "
            "with --amp-dtype fp16."
        ),
    },
}

# Provider order in price selectors; the RunPod tiers are where the
# performance was measured, the others transfer those measurements.
PROVIDER_ORDER = [
    "RunPod Secure",
    "RunPod Community",
    "Vast.ai",
    "Lambda",
    "AWS",
    "Google Cloud",
]


def optional_float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def optional_int(value: str) -> int | None:
    return int(value) if value not in ("", None) else None


def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    """Read the catalog CSV and parse the fields the summaries use."""

    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "gpu": row["gpu_name"],
                    "gpu_uuid": row["gpu_uuid"],
                    "seed": int(row["seed"]),
                    "model": row["model_key"],
                    "model_label": row["model_label"],
                    "parameters": optional_int(row["actual_unique_parameters"]),
                    "context": int(row["sequence_length"]),
                    "batch": optional_int(row["batch_size"]),
                    "status": row["status"],
                    "stable": row["stable"] == "True",
                    # Rows exported before --amp-dtype existed are all BF16.
                    "amp_dtype": row.get("amp_dtype") or "bf16",
                    "tokens_per_second": optional_float(row["tokens_per_second"]),
                    "step_ms": optional_float(row["median_step_time_ms"]),
                    "mfu_pct": optional_float(row["model_flops_utilization_pct"]),
                    "configured_mfu_pct": optional_float(
                        row["configured_clock_model_flops_utilization_pct"]
                    ),
                    "peak_tflops": optional_float(row["dense_bf16_peak_tflops"]),
                    "configured_peak_tflops": optional_float(
                        row["configured_clock_dense_bf16_peak_tflops"]
                    ),
                    "allocated_gb": optional_float(row["peak_allocated_gb"]),
                    "allocated_pct": optional_float(row["peak_allocated_vram_pct"]),
                    "reserved_gb": optional_float(row["peak_reserved_gb"]),
                    "stop_reason": row["sweep_stop_reason"],
                }
            )
    return rows


def median_of(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return statistics.median(values) if values else None


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the selection rule in the module docstring to one cell's rows."""

    seeds_run = sorted({row["seed"] for row in rows})
    complete = [row for row in rows if row["status"] == "complete" and row["stable"]]
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_batch[row["batch"]].append(row)

    batches = []
    for batch, batch_rows in sorted(by_batch.items()):
        seeds = sorted(row["seed"] for row in batch_rows)
        batches.append(
            {
                "batch": batch,
                "seeds": seeds,
                "shared": seeds == seeds_run,
                "tokens_per_second": median_of(batch_rows, "tokens_per_second"),
                "step_ms": median_of(batch_rows, "step_ms"),
                "mfu_pct": median_of(batch_rows, "mfu_pct"),
                "configured_mfu_pct": median_of(batch_rows, "configured_mfu_pct"),
                "allocated_gb": median_of(batch_rows, "allocated_gb"),
                "allocated_pct": median_of(batch_rows, "allocated_pct"),
                "reserved_gb": median_of(batch_rows, "reserved_gb"),
            }
        )

    oom_seeds: dict[int, list[int]] = defaultdict(list)
    build_oom = False
    for row in rows:
        if row["status"] == "oom_during_model_build":
            build_oom = True
        elif row["status"] == "oom":
            oom_seeds[row["batch"]].append(row["seed"])
    oom_batches = [
        {"batch": batch, "seeds": sorted(seeds)}
        for batch, seeds in sorted(oom_seeds.items())
    ]
    # A batch that completed for one seed and ran out of memory for another is
    # at the memory edge; it is reported but never selected.
    edge_batches = sorted(
        batch
        for batch, seeds in oom_seeds.items()
        if any(item["batch"] == batch for item in batches)
    )

    # No fallback to the other batches: a cell where no batch fit every seed
    # has no headline number.
    selected = max(
        (item for item in batches if item["shared"]),
        key=lambda item: (item["tokens_per_second"], item["batch"]),
        default=None,
    )
    largest_attempted = max(
        (row["batch"] for row in rows if row["batch"] is not None), default=None
    )
    saturated = any(row["stop_reason"].startswith("saturation") for row in rows)
    search_capped = (
        selected is not None
        and selected["batch"] == largest_attempted
        and largest_attempted not in oom_seeds
        and not saturated
    )
    if selected is not None:
        status = "measured"
    elif oom_batches or build_oom:
        status = "oom"
    else:
        status = "incomplete"
    return {
        "status": status,
        "seeds_run": seeds_run,
        "selected": selected,
        "batches": batches,
        "oom_batches": oom_batches,
        "oom_during_model_build": build_oom,
        "first_oom_batch": min(oom_seeds) if oom_seeds else None,
        "edge_batches": edge_batches,
        "search_capped": search_capped,
    }


def summarize_catalog(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Summarize every (GPU, model, context) cell that has catalog rows."""

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["gpu"], row["model"], row["context"])].append(row)
    return {key: summarize_cell(cell_rows) for key, cell_rows in grouped.items()}


def gpu_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-GPU MFU denominators and autocast dtype, checked for consistency."""

    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["peak_tflops"] is None:
            continue
        entry = {
            "peak_tflops": row["peak_tflops"],
            "configured_peak_tflops": row["configured_peak_tflops"],
            "amp_dtype": row["amp_dtype"],
        }
        previous = metadata.setdefault(row["gpu"], entry)
        if previous != entry:
            raise ValueError(f"inconsistent MFU denominators for {row['gpu']}")
    return metadata


def model_parameters(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Measured unique parameter count and label for every model key."""

    models: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["parameters"] is None:
            continue
        entry = {"label": row["model_label"], "parameters": row["parameters"]}
        previous = models.setdefault(row["model"], entry)
        if previous != entry:
            raise ValueError(f"inconsistent parameter count for {row['model']}")
    return {key: models[key] for key in MODEL_ORDER if key in models}


def load_price_profiles(
    runpod_path: Path = RUNPOD_PRICING_PATH,
    provider_path: Path = PROVIDER_PRICING_PATH,
) -> dict[str, Any]:
    """Join both price snapshots into one quote table per provider.

    RunPod Secure/Community rates come from the RunPod snapshot and are checked
    against the RunPod quotes in the multi-provider snapshot.
    """

    runpod = json.loads(runpod_path.read_text())
    providers_snapshot = json.loads(provider_path.read_text())
    profiles: dict[str, dict[str, Any]] = {
        name: {"quotes": {}} for name in PROVIDER_ORDER
    }
    for gpu, price in runpod["gpus"].items():
        for tier, field in (
            ("RunPod Secure", "secure_usd_per_hour"),
            ("RunPod Community", "community_usd_per_hour"),
        ):
            profiles[tier]["quotes"][gpu] = {
                "usd_per_gpu_hour": price[field],
                "product": price["display_name"],
                "price_type": "on_demand",
                "match_quality": "exact",
                "instance_gpu_count": 1,
                "notes": "",
                "source_url": runpod["public_source"],
            }
    for quote in providers_snapshot["quotes"]:
        provider = quote["provider"]
        if provider not in profiles:
            raise ValueError(f"unknown provider in price snapshot: {provider}")
        if provider.startswith("RunPod"):
            expected = profiles[provider]["quotes"][quote["gpu"]]["usd_per_gpu_hour"]
            if abs(expected - quote["usd_per_gpu_hour"]) > 1e-9:
                raise ValueError(
                    f"{provider} {quote['gpu']}: snapshots disagree "
                    f"({expected} vs {quote['usd_per_gpu_hour']})"
                )
            # Keep the provider snapshot's price type (community is a
            # marketplace), but the RunPod snapshot's product name.
            profiles[provider]["quotes"][quote["gpu"]]["price_type"] = quote[
                "price_type"
            ]
            continue
        profiles[provider]["quotes"][quote["gpu"]] = {
            key: quote[key]
            for key in (
                "usd_per_gpu_hour",
                "product",
                "price_type",
                "match_quality",
                "instance_gpu_count",
                "notes",
                "source_url",
            )
        }
    return {
        "runpod": runpod,
        "providers": providers_snapshot,
        "profiles": profiles,
    }


def gpu_profile(gpu: str, runpod: dict[str, Any]) -> dict[str, Any]:
    """Display name, VRAM and RunPod Secure rate (``None`` if unpriced)."""

    if gpu in runpod["gpus"]:
        price = runpod["gpus"][gpu]
        return {
            "display_name": price["display_name"],
            "vram_gb": price["vram_gb"],
            "secure_usd_per_hour": price["secure_usd_per_hour"],
        }
    info = NOT_YET_MEASURED.get(gpu, {})
    return {
        "display_name": info.get("display_name", gpu.removeprefix("NVIDIA ")),
        "vram_gb": info.get("vram_gb"),
        "secure_usd_per_hour": None,
    }


def gpus_by_price(gpus: Any, runpod: dict[str, Any]) -> list[str]:
    """Order GPUs by RunPod Secure rate; GPUs RunPod does not rent go last."""

    def key(gpu: str) -> tuple[float, str]:
        rate = gpu_profile(gpu, runpod)["secure_usd_per_hour"]
        return (rate if rate is not None else float("inf"), gpu)

    return sorted(gpus, key=key)


def usd_per_billion_tokens(tokens_per_second: float, usd_per_hour: float) -> float:
    return 1e9 / tokens_per_second / 3600.0 * usd_per_hour


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
