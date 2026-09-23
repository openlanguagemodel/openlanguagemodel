#!/usr/bin/env python3
"""Export context-frontier and reused 2,048-token raw rows to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from annotate_clock_adjusted_mfu import (
    GPU_REFERENCES,
    configured_max_sm_clock,
    registered_reference_clock,
)


FIELDS = [
    "source_file",
    "source_kind",
    "benchmark",
    "gpu_name",
    "gpu_uuid",
    "seed",
    "model_key",
    "model_label",
    "actual_unique_parameters",
    "sequence_length",
    "batch_size",
    "tokens_per_step",
    "status",
    "model_status",
    "context_status",
    "stable",
    "is_center_batch",
    "is_best_batch_at_context",
    "is_max_tokens_per_second_across_contexts",
    "is_max_mfu_across_contexts",
    "candidate_batch_sizes",
    "warmup_steps_executed",
    "measured_steps",
    "median_step_time_ms",
    "mean_step_time_ms",
    "p25_step_time_ms",
    "p75_step_time_ms",
    "measured_robust_relative_jitter",
    "tokens_per_second",
    "mean_based_tokens_per_second",
    "training_flops_per_step",
    "achieved_tflops",
    "dense_bf16_peak_tflops",
    "model_flops_utilization_pct",
    "configured_clock_dense_bf16_peak_tflops",
    "configured_clock_model_flops_utilization_pct",
    "peak_allocated_bytes",
    "peak_allocated_gb",
    "peak_allocated_vram_pct",
    "peak_reserved_bytes",
    "peak_reserved_gb",
    "peak_reserved_vram_pct",
    "build_peak_allocated_gb",
    "build_peak_reserved_gb",
    "build_seconds",
    "build_wall_seconds",
    "selected_sdpa_backend",
    "sweep_stop_reason",
    "error",
]


def gpu_uuid(environment: dict[str, Any]) -> str:
    fields = [
        field.strip() for field in (environment.get("nvidia_smi") or "").split(",")
    ]
    return fields[1] if len(fields) > 1 else ""


def batch_fields(batch: dict[str, Any]) -> dict[str, Any]:
    return {field: batch[field] for field in FIELDS if field in batch}


def rows_for_frontier(path: Path) -> list[dict[str, Any]]:
    frontier = json.loads(path.read_text())
    if frontier.get("benchmark") != "olm_single_gpu_context_batch_frontier":
        raise ValueError(f"{path}: wrong benchmark type")
    if frontier.get("status") != "complete":
        raise ValueError(f"{path}: frontier is incomplete")

    baseline_path = Path(frontier["baseline"]["source"])
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("status") != "complete":
        raise ValueError(f"{baseline_path}: baseline is incomplete")
    if baseline["seed"] != frontier["seed"]:
        raise ValueError(f"{path}: baseline seed mismatch")

    environment = frontier["environment"]
    reference = GPU_REFERENCES[environment["device_name"]]
    configured_clock = configured_max_sm_clock(environment)
    configured_peak = (
        reference["nominal_dense_bf16_peak_tflops"]
        * configured_clock
        / registered_reference_clock(reference)
    )
    common = {
        "gpu_name": environment["device_name"],
        "gpu_uuid": gpu_uuid(environment),
        "seed": frontier["seed"],
        "dense_bf16_peak_tflops": environment["dense_bf16_peak_tflops"],
        "configured_clock_dense_bf16_peak_tflops": configured_peak,
    }
    rows: list[dict[str, Any]] = []

    for model in baseline["models"]:
        batches = model.get("batches", [])
        if not batches:
            rows.append(
                {
                    **common,
                    "source_file": baseline_path.name,
                    "source_kind": "baseline_reused",
                    "benchmark": baseline["benchmark"],
                    "model_key": model["key"],
                    "model_label": model["config"]["label"],
                    "actual_unique_parameters": model.get(
                        "actual_unique_parameters"
                    ),
                    "sequence_length": baseline["sequence_length"],
                    "batch_size": "",
                    "status": model["status"],
                    "model_status": model["status"],
                    "context_status": model["status"],
                    "is_center_batch": False,
                    "is_best_batch_at_context": False,
                    "candidate_batch_sizes": json.dumps(
                        baseline["batch_sizes"],
                        separators=(",", ":"),
                    ),
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
                **common,
                "source_file": baseline_path.name,
                "source_kind": "baseline_reused",
                "benchmark": baseline["benchmark"],
                "model_key": model["key"],
                "model_label": model["config"]["label"],
                "actual_unique_parameters": model.get(
                    "actual_unique_parameters"
                ),
                "sequence_length": baseline["sequence_length"],
                "batch_size": batch["batch_size"],
                "model_status": model["status"],
                "context_status": model["status"],
                "is_center_batch": (
                    batch["batch_size"] == model.get("best_batch_size")
                ),
                "is_best_batch_at_context": (
                    batch["status"] == "complete"
                    and batch["batch_size"] == model.get("best_batch_size")
                ),
                "candidate_batch_sizes": json.dumps(
                    baseline["batch_sizes"],
                    separators=(",", ":"),
                ),
                "sweep_stop_reason": model.get("sweep_stop_reason"),
                **batch_fields(batch),
            }
            if batch["status"] == "complete":
                row["configured_clock_model_flops_utilization_pct"] = (
                    100.0 * batch["achieved_tflops"] / configured_peak
                )
            rows.append(row)

    for model in frontier["models"]:
        for context in model["contexts"]:
            batches = context["batches"]
            if not batches:
                rows.append(
                    {
                        **common,
                        "source_file": path.name,
                        "source_kind": "context_frontier",
                        "benchmark": frontier["benchmark"],
                        "model_key": model["key"],
                        "model_label": model["config"]["label"],
                        "actual_unique_parameters": model.get(
                            "actual_unique_parameters"
                        ),
                        "sequence_length": context["sequence_length"],
                        "batch_size": "",
                        "status": context["status"],
                        "model_status": model.get("status", "complete"),
                        "context_status": context["status"],
                        "is_center_batch": False,
                        "is_best_batch_at_context": False,
                        "candidate_batch_sizes": json.dumps(
                            context["candidate_batch_sizes"],
                            separators=(",", ":"),
                        ),
                        "build_peak_allocated_gb": context.get(
                            "build_peak_allocated_gb"
                        ),
                        "build_peak_reserved_gb": context.get(
                            "build_peak_reserved_gb"
                        ),
                        "build_wall_seconds": context.get(
                            "build_wall_seconds"
                        ),
                        "sweep_stop_reason": context.get(
                            "sweep_stop_reason"
                        ),
                        "error": context.get("error"),
                    }
                )
                continue
            for batch in batches:
                row = {
                    **common,
                    "source_file": path.name,
                    "source_kind": "context_frontier",
                    "benchmark": frontier["benchmark"],
                    "model_key": model["key"],
                    "model_label": model["config"]["label"],
                    "actual_unique_parameters": model[
                        "actual_unique_parameters"
                    ],
                    "sequence_length": context["sequence_length"],
                    "batch_size": batch["batch_size"],
                    "model_status": model.get("status", "complete"),
                    "context_status": context["status"],
                    "is_center_batch": (
                        batch["batch_size"] == context["center_batch_size"]
                    ),
                    "is_best_batch_at_context": (
                        batch["status"] == "complete"
                        and batch["batch_size"]
                        == context.get("best_batch_size")
                    ),
                    "candidate_batch_sizes": json.dumps(
                        context["candidate_batch_sizes"],
                        separators=(",", ":"),
                    ),
                    "sweep_stop_reason": context.get("sweep_stop_reason"),
                    **batch_fields(batch),
                }
                if batch["status"] == "complete":
                    row["configured_clock_model_flops_utilization_pct"] = (
                        100.0 * batch["achieved_tflops"] / configured_peak
                    )
                rows.append(row)

    return rows


def merge_and_mark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["gpu_uuid"],
            row["seed"],
            row["model_key"],
            row["sequence_length"],
            row["batch_size"],
            row["source_kind"],
        )
        previous = deduplicated.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting duplicate row: {key}")
        deduplicated[key] = row
    merged = list(deduplicated.values())

    successful_by_model: dict[
        tuple[str, int, str],
        list[dict[str, Any]],
    ] = {}
    for row in merged:
        row["is_max_tokens_per_second_across_contexts"] = False
        row["is_max_mfu_across_contexts"] = False
        if row["status"] == "complete":
            key = (row["gpu_uuid"], row["seed"], row["model_key"])
            successful_by_model.setdefault(key, []).append(row)
    for model_rows in successful_by_model.values():
        max_tps = max(row["tokens_per_second"] for row in model_rows)
        max_mfu = max(
            row["model_flops_utilization_pct"] for row in model_rows
        )
        for row in model_rows:
            row["is_max_tokens_per_second_across_contexts"] = (
                row["tokens_per_second"] == max_tps
            )
            row["is_max_mfu_across_contexts"] = (
                row["model_flops_utilization_pct"] == max_mfu
            )
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["gpu_uuid"],
            row["seed"],
            row["model_key"],
            row["sequence_length"],
        )
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        best = [
            row for row in group if row.get("is_best_batch_at_context")
        ]
        if best:
            selected.extend(best)
        else:
            # Preserve an explicit OOM row when no batch fit this context.
            selected.append(
                min(
                    group,
                    key=lambda row: (
                        int(row["batch_size"])
                        if row.get("batch_size") not in ("", None)
                        else -1
                    ),
                )
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        rows.extend(rows_for_frontier(input_path))
    rows = merge_and_mark(rows)
    rows.sort(
        key=lambda row: (
            row["gpu_name"],
            row["gpu_uuid"],
            row["seed"],
            row["model_key"],
            row["sequence_length"],
            (
                int(row["batch_size"])
                if row.get("batch_size") not in ("", None)
                else -1
            ),
        )
    )
    selected = selected_rows(rows)
    write_csv(args.all_output, rows)
    write_csv(args.selected_output, selected)
    print(
        f"Wrote {len(rows)} non-aggregated rows to {args.all_output}; "
        f"{len(selected)} selected-context rows to {args.selected_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
