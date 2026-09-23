#!/usr/bin/env python3
"""Look up the best measured batch for a GPU/model/context training point."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CATALOG = (
    Path(__file__).resolve().parent.parent / "data" / "benchmark_catalog.csv"
)


def normalized(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def numeric(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if not value:
        raise ValueError(f"row has no {field}: {row}")
    return float(value)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def select_gpu(rows: list[dict[str, str]], query: str) -> str:
    names = sorted({row["gpu_name"] for row in rows})
    query_key = normalized(query)
    matches = [
        name
        for name in names
        if query_key in normalized(name) or normalized(name) in query_key
    ]
    if not matches:
        raise ValueError(f"GPU {query!r} not found; available: {names}")
    if len(matches) > 1:
        raise ValueError(f"GPU query {query!r} is ambiguous: {matches}")
    return matches[0]


def best_batch(
    groups: dict[int, list[dict[str, str]]],
    field: str,
) -> int:
    return max(
        groups,
        key=lambda batch: statistics.median(
            numeric(row, field) for row in groups[batch]
        ),
    )


def lookup(
    catalog: Path,
    gpu_query: str,
    model_query: str,
    context_length: int,
    gpu_uuid: str | None,
) -> dict[str, Any]:
    with catalog.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{catalog} is empty")

    gpu_name = select_gpu(rows, gpu_query)
    model_key = normalized(model_query)
    matching = [
        row
        for row in rows
        if row["gpu_name"] == gpu_name
        and normalized(row["model_key"]) == model_key
        and int(row["sequence_length"]) == context_length
    ]
    if gpu_uuid is not None:
        matching = [row for row in matching if row["gpu_uuid"] == gpu_uuid]
    uuids = sorted({row["gpu_uuid"] for row in matching})
    if not matching:
        contexts = sorted(
            {
                int(row["sequence_length"])
                for row in rows
                if row["gpu_name"] == gpu_name
                and normalized(row["model_key"]) == model_key
            }
        )
        raise ValueError(
            f"no result for {gpu_name}, {model_query}, S={context_length}; "
            f"available contexts: {contexts}"
        )
    if len(uuids) > 1:
        raise ValueError(
            f"multiple physical GPUs match; choose --gpu-uuid from {uuids}"
        )

    successful = [row for row in matching if row["status"] == "complete"]
    if not successful:
        attempted_batches = sorted(
            {
                int(row["batch_size"])
                for row in matching
                if row.get("batch_size") not in ("", None)
            }
        )
        model_build_oom = any(
            row["status"] == "oom_during_model_build"
            for row in matching
        )
        return {
            "gpu_name": gpu_name,
            "gpu_uuid": uuids[0] if uuids else None,
            "model_key": model_key,
            "sequence_length": context_length,
            "status": (
                "oom_during_model_build"
                if model_build_oom
                else "no_successful_batch"
            ),
            "attempted_batches": attempted_batches,
        }

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in successful:
        groups[int(row["batch_size"])].append(row)

    tps_batch = best_batch(groups, "tokens_per_second")
    mfu_batch = best_batch(
        groups,
        "configured_clock_model_flops_utilization_pct",
    )
    selected_batch = tps_batch
    selected_rows = sorted(
        groups[selected_batch],
        key=lambda row: int(row["seed"]),
    )
    seeds = [int(row["seed"]) for row in selected_rows]
    tokens_per_second = [
        numeric(row, "tokens_per_second") for row in selected_rows
    ]
    configured_mfu = [
        numeric(row, "configured_clock_model_flops_utilization_pct")
        for row in selected_rows
    ]
    nominal_mfu = [
        numeric(row, "model_flops_utilization_pct") for row in selected_rows
    ]
    peak_vram_gb = [
        numeric(row, "peak_allocated_gb") for row in selected_rows
    ]
    peak_vram_pct = [
        numeric(row, "peak_allocated_vram_pct") for row in selected_rows
    ]
    oom_batches = sorted(
        {
            int(row["batch_size"])
            for row in matching
            if row["status"] == "oom"
        }
    )

    return {
        "gpu_name": gpu_name,
        "gpu_uuid": uuids[0],
        "model_key": selected_rows[0]["model_key"],
        "model_label": selected_rows[0]["model_label"],
        "actual_unique_parameters": int(
            selected_rows[0]["actual_unique_parameters"]
        ),
        "sequence_length": context_length,
        "status": "complete",
        "best_batch_for_tokens_per_second": tps_batch,
        "best_batch_for_configured_clock_mfu": mfu_batch,
        "same_best_batch_for_both_objectives": tps_batch == mfu_batch,
        "tokens_per_step": int(selected_rows[0]["tokens_per_step"]),
        "replicate_seeds": seeds,
        "tokens_per_second": summarize(tokens_per_second),
        "configured_clock_mfu_pct": summarize(configured_mfu),
        "nominal_reference_mfu_pct": summarize(nominal_mfu),
        "peak_allocated_vram_gb": summarize(peak_vram_gb),
        "peak_allocated_vram_pct": summarize(peak_vram_pct),
        "configured_clock_dense_bf16_peak_tflops": numeric(
            selected_rows[0],
            "configured_clock_dense_bf16_peak_tflops",
        ),
        "first_measured_oom_batch": min(oom_batches) if oom_batches else None,
        "replicates": [
            {
                "seed": int(row["seed"]),
                "tokens_per_second": numeric(row, "tokens_per_second"),
                "configured_clock_mfu_pct": numeric(
                    row,
                    "configured_clock_model_flops_utilization_pct",
                ),
                "nominal_reference_mfu_pct": numeric(
                    row,
                    "model_flops_utilization_pct",
                ),
                "peak_allocated_vram_gb": numeric(
                    row,
                    "peak_allocated_gb",
                ),
                "peak_allocated_vram_pct": numeric(
                    row,
                    "peak_allocated_vram_pct",
                ),
            }
            for row in selected_rows
        ],
    }


def markdown(result: dict[str, Any]) -> str:
    if result["status"] != "complete":
        if result["status"] == "oom_during_model_build":
            return (
                f"{result['gpu_name']} · {result['model_key']} · "
                f"S={result['sequence_length']}: OOM during model "
                "construction; no batch was attempted."
            )
        return (
            f"{result['gpu_name']} · {result['model_key']} · "
            f"S={result['sequence_length']}: no successful batch; "
            f"attempted {result['attempted_batches']}"
        )

    lines = [
        (
            f"{result['gpu_name']} · {result['model_label']} · "
            f"context {result['sequence_length']:,}"
        ),
        "",
        (
            f"Best batch: {result['best_batch_for_tokens_per_second']} "
            f"({result['tokens_per_step']:,} tokens/step). "
            f"TPS and MFU select the same batch: "
            f"{result['same_best_batch_for_both_objectives']}."
        ),
        "",
        "| Seed | Tokens/s | Configured-clock MFU | Peak allocated VRAM |",
        "|---:|---:|---:|---:|",
    ]
    for replicate in result["replicates"]:
        lines.append(
            f"| {replicate['seed']} "
            f"| {replicate['tokens_per_second']:,.1f} "
            f"| {replicate['configured_clock_mfu_pct']:.2f}% "
            f"| {replicate['peak_allocated_vram_gb']:.2f} GB "
            f"({replicate['peak_allocated_vram_pct']:.2f}%) |"
        )
    lines.extend(
        [
            "",
            (
                f"Expected (replicate median): "
                f"{result['tokens_per_second']['median']:,.1f} tokens/s, "
                f"{result['configured_clock_mfu_pct']['median']:.2f}% MFU."
            ),
        ]
    )
    if result["first_measured_oom_batch"] is not None:
        lines.append(
            f"First measured OOM batch: "
            f"{result['first_measured_oom_batch']}."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    args = parser.parse_args()

    result = lookup(
        catalog=args.catalog,
        gpu_query=args.gpu,
        model_query=args.model,
        context_length=args.context,
        gpu_uuid=args.gpu_uuid,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
