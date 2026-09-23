#!/usr/bin/env python3
"""Flatten raw OLM training benchmark JSON files into non-aggregated CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "source_file",
    "benchmark",
    "gpu_name",
    "gpu_uuid",
    "seed",
    "model_key",
    "model_label",
    "actual_unique_parameters",
    "sequence_length",
    "batch_size",
    "status",
    "model_status",
    "stable",
    "is_best_batch_for_seed",
    "warmup_steps_executed",
    "measured_steps",
    "median_step_time_ms",
    "mean_step_time_ms",
    "p25_step_time_ms",
    "p75_step_time_ms",
    "measured_robust_relative_jitter",
    "tokens_per_step",
    "tokens_per_second",
    "mean_based_tokens_per_second",
    "training_flops_per_step",
    "achieved_tflops",
    "dense_bf16_peak_tflops",
    "model_flops_utilization_pct",
    "peak_allocated_bytes",
    "peak_allocated_gb",
    "peak_allocated_vram_pct",
    "peak_reserved_bytes",
    "peak_reserved_gb",
    "peak_reserved_vram_pct",
    "build_peak_allocated_gb",
    "build_peak_reserved_gb",
    "build_seconds",
    "sweep_stop_reason",
    "error",
]


def gpu_uuid(environment: dict[str, Any]) -> str:
    # nvidia_smi is retained as raw CSV metadata. Its second field is the UUID.
    fields = [
        field.strip() for field in (environment.get("nvidia_smi") or "").split(",")
    ]
    return fields[1] if len(fields) > 1 else ""


def rows_for(path: Path) -> list[dict[str, Any]]:
    record = json.loads(path.read_text())
    if record.get("benchmark") != "olm_single_gpu_training_mfu":
        raise ValueError(f"{path}: not an OLM training-MFU result")
    if record.get("status") != "complete":
        raise ValueError(f"{path}: result is not complete")

    environment = record["environment"]
    rows: list[dict[str, Any]] = []
    for model in record["models"]:
        best_batch = model.get("best_batch_size")
        batches = model.get("batches", [])
        if not batches:
            rows.append(
                {
                    "source_file": path.name,
                    "benchmark": record["benchmark"],
                    "gpu_name": environment["device_name"],
                    "gpu_uuid": gpu_uuid(environment),
                    "seed": record["seed"],
                    "model_key": model["key"],
                    "model_label": model["config"]["label"],
                    "actual_unique_parameters": model.get(
                        "actual_unique_parameters"
                    ),
                    "sequence_length": record["sequence_length"],
                    "batch_size": "",
                    "status": model["status"],
                    "model_status": model["status"],
                    "is_best_batch_for_seed": False,
                    "build_peak_allocated_gb": model.get(
                        "build_peak_allocated_gb"
                    ),
                    "build_peak_reserved_gb": model.get(
                        "build_peak_reserved_gb"
                    ),
                    "build_seconds": model.get("build_seconds"),
                    "sweep_stop_reason": model.get("sweep_stop_reason"),
                    "error": model.get("error"),
                }
            )
            continue
        for batch in batches:
            row = {
                "source_file": path.name,
                "benchmark": record["benchmark"],
                "gpu_name": environment["device_name"],
                "gpu_uuid": gpu_uuid(environment),
                "seed": record["seed"],
                "model_key": model["key"],
                "model_label": model["config"]["label"],
                "actual_unique_parameters": model.get("actual_unique_parameters"),
                "sequence_length": record["sequence_length"],
                "batch_size": batch["batch_size"],
                "status": batch["status"],
                "model_status": model["status"],
                "stable": batch.get("stable"),
                "is_best_batch_for_seed": (
                    batch["status"] == "complete"
                    and batch["batch_size"] == best_batch
                ),
                "sweep_stop_reason": model.get("sweep_stop_reason"),
            }
            for field in FIELDS:
                if field not in row and field in batch:
                    row[field] = batch[field]
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        rows.extend(rows_for(input_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"Wrote {len(rows)} non-aggregated rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
