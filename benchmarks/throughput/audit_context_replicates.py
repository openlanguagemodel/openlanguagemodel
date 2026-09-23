#!/usr/bin/env python3
"""Audit context-frontier replicates without requiring identical OOM boundaries.

The strict validator intentionally rejects replicate sets whose success/OOM
frontier differs.  This companion audit preserves those differences as
machine-readable evidence while summarizing variation only where every seed
completed the same coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "tokens_per_second",
    "model_flops_utilization_pct",
    "peak_allocated_gb",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("benchmark") != "olm_single_gpu_context_batch_frontier":
        raise ValueError(f"{path}: wrong benchmark type")
    if record.get("status") != "complete":
        raise ValueError(f"{path}: result is incomplete")
    return record


def flatten(record: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for model in record["models"]:
        for context in model["contexts"]:
            for batch in context["batches"]:
                coordinate = (
                    model["key"],
                    context["sequence_length"],
                    batch["batch_size"],
                )
                if coordinate in rows:
                    raise ValueError(f"duplicate coordinate: {coordinate}")
                rows[coordinate] = batch
    return rows


def metric_summary(
    coordinate_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = {
            str(seed): row[metric]
            for seed, row in sorted(coordinate_rows.items())
        }
        numeric_values = list(values.values())
        mean = statistics.mean(numeric_values)
        summary[metric] = {
            "by_seed": values,
            "mean": mean,
            "min": min(numeric_values),
            "max": max(numeric_values),
            "relative_range": (
                (max(numeric_values) - min(numeric_values)) / mean
                if mean
                else 0.0
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least two replicate files are required")

    loaded = [(path, load(path)) for path in args.inputs]
    seeds = [record["seed"] for _, record in loaded]
    if len(set(seeds)) != len(seeds):
        parser.error("replicate seeds must be unique")

    gpu_names = {
        record["environment"]["device_name"] for _, record in loaded
    }
    gpu_uuids = {
        (record["environment"].get("nvidia_smi") or "").split(",")[1].strip()
        for _, record in loaded
    }
    if len(gpu_names) != 1 or len(gpu_uuids) != 1:
        parser.error("all replicates must use the same physical GPU")

    flattened = {
        record["seed"]: flatten(record) for _, record in loaded
    }
    coordinates = sorted(
        set().union(*(set(rows) for rows in flattened.values()))
    )
    status_differences: list[dict[str, Any]] = []
    common_complete: list[dict[str, Any]] = []

    for model_key, sequence_length, batch_size in coordinates:
        coordinate = (model_key, sequence_length, batch_size)
        by_seed = {
            seed: rows.get(coordinate) for seed, rows in flattened.items()
        }
        statuses = {
            str(seed): (
                row["status"] if row is not None else "not_attempted"
            )
            for seed, row in sorted(by_seed.items())
        }
        if len(set(statuses.values())) != 1:
            status_differences.append(
                {
                    "model_key": model_key,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "status_by_seed": statuses,
                }
            )
            continue
        if next(iter(statuses.values())) == "complete":
            complete_rows = {
                seed: row
                for seed, row in by_seed.items()
                if row is not None
            }
            common_complete.append(
                {
                    "model_key": model_key,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "metrics": metric_summary(complete_rows),
                }
            )

    record = {
        "schema_version": 1,
        "benchmark": "olm_context_frontier_replicate_audit",
        "gpu_name": next(iter(gpu_names)),
        "gpu_uuid": next(iter(gpu_uuids)),
        "seeds": sorted(seeds),
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "seed": result["seed"],
            }
            for path, result in loaded
        ],
        "coordinate_count": len(coordinates),
        "common_complete_coordinate_count": len(common_complete),
        "frontier_identical": not status_differences,
        "status_difference_count": len(status_differences),
        "status_differences": status_differences,
        "common_complete_coordinates": common_complete,
        "note": (
            "This is a derived audit. Source JSON measurements are unchanged. "
            "A not_attempted status means the ascending batch sweep stopped "
            "after an earlier out-of-memory result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"Wrote {args.output}: coordinates={len(coordinates)}, "
        f"common_complete={len(common_complete)}, "
        f"status_differences={len(status_differences)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
