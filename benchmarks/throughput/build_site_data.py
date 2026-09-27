#!/usr/bin/env python3
"""Build the compact JSON dataset behind the website's /benchmarks page.

The page never contains hand-typed measurements. This script reads the
canonical catalog, both price snapshots and the multi-GPU scaling summary,
applies the cell-selection rule in ``catalog.py``, and writes one
deterministic JSON file (no timestamps, sorted keys) that the Next.js build
imports. Run it whenever a data file changes; ``--check`` exits non-zero when
the committed JSON is stale, which is how the test suite keeps it honest.

    python benchmarks/throughput/build_site_data.py
    python benchmarks/throughput/build_site_data.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from catalog import (
    CATALOG_PATH,
    NOT_YET_MEASURED,
    PROTOCOL_SEEDS,
    PROVIDER_ORDER,
    SCALING_PATH,
    gpu_metadata,
    load_catalog,
    load_price_profiles,
    model_parameters,
    sha256_of,
    summarize_catalog,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPOSITORY_ROOT / "website" / "app" / "lib" / "benchmarks.generated.json"
SIZE_BUDGET_BYTES = 300_000


def rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def compact_batch(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch": item["batch"],
        "seeds": item["seeds"],
        "tps": rounded(item["tokens_per_second"], 1),
        "stepMs": rounded(item["step_ms"], 2),
        "mfu": rounded(item["mfu_pct"], 3),
        "mfuBoard": rounded(item["configured_mfu_pct"], 3),
        "vramGb": rounded(item["allocated_gb"], 3),
        "vramPct": rounded(item["allocated_pct"], 2),
        "reservedGb": rounded(item["reserved_gb"], 3),
    }


def build_payload() -> dict[str, Any]:
    rows = load_catalog()
    cells = summarize_catalog(rows)
    denominators = gpu_metadata(rows)
    models = model_parameters(rows)
    prices = load_price_profiles()
    runpod = prices["runpod"]
    providers = prices["providers"]
    scaling = json.loads(SCALING_PATH.read_text())

    measured_gpus = sorted(
        denominators,
        key=lambda gpu: runpod["gpus"][gpu]["secure_usd_per_hour"],
    )
    gpus = []
    for gpu in measured_gpus:
        price = runpod["gpus"][gpu]
        gpus.append(
            {
                "id": gpu,
                "name": price["display_name"],
                "vramGb": price["vram_gb"],
                "ampDtype": denominators[gpu]["amp_dtype"],
                "peakTflops": rounded(denominators[gpu]["peak_tflops"], 3),
                "boardPeakTflops": rounded(
                    denominators[gpu]["configured_peak_tflops"], 3
                ),
            }
        )

    pending = [
        {
            "id": gpu,
            "name": info["display_name"],
            "vramGb": info["vram_gb"],
            "status": "not_yet_measured",
            "note": info["note"],
        }
        for gpu, info in NOT_YET_MEASURED.items()
    ]
    pending.extend(
        {
            "id": gpu,
            "name": price["display_name"],
            "vramGb": price["vram_gb"],
            "status": "not_in_catalog",
            "note": "Priced in the RunPod snapshot; no rows in the catalog yet.",
        }
        for gpu, price in sorted(
            runpod["gpus"].items(), key=lambda item: item[1]["secure_usd_per_hour"]
        )
        if gpu not in denominators
    )

    cell_list = []
    for (gpu, model, context), cell in sorted(
        cells.items(),
        key=lambda item: (
            measured_gpus.index(item[0][0]),
            list(models).index(item[0][1]),
            item[0][2],
        ),
    ):
        cell_list.append(
            {
                "gpu": gpu,
                "model": model,
                "context": context,
                "status": cell["status"],
                "seedsRun": cell["seeds_run"],
                "selectedBatch": (
                    cell["selected"]["batch"] if cell["selected"] else None
                ),
                "batches": [compact_batch(item) for item in cell["batches"]],
                "oom": cell["oom_batches"],
                "buildOom": cell["oom_during_model_build"],
                "edgeBatches": cell["edge_batches"],
                "searchCapped": cell["search_capped"],
            }
        )

    provider_list = []
    for name in PROVIDER_ORDER:
        profile = prices["profiles"][name]
        measured_host = name.startswith("RunPod")
        provider_list.append(
            {
                "id": name.lower().replace(" ", "-").replace(".", ""),
                "label": name,
                "measuredHost": measured_host,
                "capturedAt": (
                    runpod["captured_at"] if measured_host else providers["captured_at"]
                ),
                "quotes": {
                    gpu: {
                        "usd": quote["usd_per_gpu_hour"],
                        "product": quote["product"],
                        "priceType": quote["price_type"],
                        "match": quote["match_quality"],
                        "gpusPerInstance": quote["instance_gpu_count"],
                        "notes": quote["notes"],
                        "url": quote["source_url"],
                    }
                    for gpu, quote in sorted(profile["quotes"].items())
                },
            }
        )

    return {
        "schemaVersion": 1,
        "source": {
            "generator": relative(Path(__file__)),
            "catalog": relative(CATALOG_PATH),
            "catalogSha256": sha256_of(CATALOG_PATH),
            "catalogRows": len(rows),
            "runpodPricing": {
                "capturedAt": runpod["captured_at"],
                "url": runpod["public_source"],
                "notes": runpod["notes"],
            },
            "providerPricing": {
                "capturedAt": providers["captured_at"],
                "policy": providers["comparison_policy"],
            },
        },
        "protocolSeeds": list(PROTOCOL_SEEDS),
        "models": [
            {"key": key, "label": info["label"], "parameters": info["parameters"]}
            for key, info in models.items()
        ],
        "contexts": sorted({row["context"] for row in rows}),
        "gpus": gpus,
        "pendingGpus": pending,
        "cells": cell_list,
        "providers": provider_list,
        "scaling": {
            "workload": scaling["workload"],
            "source": scaling["source"],
            "rows": [
                {
                    "gpus": row["gpu_count"],
                    "replicates": len(row["replicate_tokens_per_second"]),
                    "tps": rounded(row["mean_tokens_per_second"], 1),
                    "tpsStd": rounded(row["std_tokens_per_second"], 1),
                    "efficiency": rounded(row["scaling_efficiency"], 4),
                    "peakGb": rounded(max(row["peak_allocated_gb_per_gpu"]), 3),
                }
                for row in scaling["rows"]
            ],
        },
    }


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output file differs from a fresh build.",
    )
    args = parser.parse_args()

    text = serialize(build_payload())
    size = len(text.encode())
    if size > SIZE_BUDGET_BYTES:
        raise SystemExit(f"site data is {size:,} bytes; budget {SIZE_BUDGET_BYTES:,}")
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != text:
            print(
                f"{args.output} is stale; run {Path(__file__).name}",
                file=sys.stderr,
            )
            return 1
        print(f"OK {args.output} ({size:,} bytes)")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Wrote {args.output} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
