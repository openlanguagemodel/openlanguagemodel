#!/usr/bin/env python3
"""Validate completed OLM context/batch-frontier replicate files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_wall_timestamps(
    item: dict[str, Any],
    label: str,
) -> None:
    start = item.get("wall_start_unix_ns")
    finish = item.get("wall_finish_unix_ns")
    require(
        isinstance(start, int) and isinstance(finish, int) and start < finish,
        f"{label} invalid wall-clock bounds",
    )
    require(
        isinstance(item.get("started_at"), str)
        and isinstance(item.get("finished_at"), str),
        f"{label} missing ISO timestamps",
    )


def validate_step_timestamps(
    batch: dict[str, Any],
    phase: str,
    expected_count: int,
    label: str,
) -> None:
    starts = batch.get(f"{phase}_step_wall_start_unix_ns")
    finishes = batch.get(f"{phase}_step_wall_finish_unix_ns")
    require(
        isinstance(starts, list)
        and isinstance(finishes, list)
        and len(starts) == expected_count
        and len(finishes) == expected_count,
        f"{label} wrong {phase} timestamp count",
    )
    require(
        all(
            isinstance(start, int) and isinstance(finish, int) and start < finish
            for start, finish in zip(starts, finishes)
        ),
        f"{label} invalid {phase} timestamp bounds",
    )


def point_signature(
    record: dict[str, Any],
) -> list[tuple[str, int, int, str]]:
    return [
        (
            model["key"],
            context["sequence_length"],
            batch["batch_size"],
            batch["status"],
        )
        for model in record["models"]
        for context in model["contexts"]
        for batch in context["batches"]
    ]


def candidate_signature(
    record: dict[str, Any],
) -> list[tuple[str, int, tuple[int, ...]]]:
    return [
        (
            model["key"],
            context["sequence_length"],
            tuple(context["candidate_batch_sizes"]),
        )
        for model in record["models"]
        for context in model["contexts"]
    ]


def validate(path: Path, allow_all_oom: bool = False) -> dict[str, Any]:
    record = json.loads(path.read_text())
    prefix = f"{path}:"
    require(
        record.get("benchmark") == "olm_single_gpu_context_batch_frontier",
        f"{prefix} wrong benchmark type",
    )
    require(record.get("status") == "complete", f"{prefix} suite is incomplete")
    require(record.get("suite_wall_seconds", 0) > 0, f"{prefix} invalid wall time")
    require(record.get("models"), f"{prefix} no eligible models")

    contexts = record["context_lengths_run"]
    measured_steps = record["protocol"]["measured_steps"]
    total_device_bytes = record["environment"]["total_device_bytes"]
    schema_version = record.get("schema_version", 1)
    complete_batches = 0
    oom_batches = 0
    for model in record["models"]:
        require(
            [context["sequence_length"] for context in model["contexts"]] == contexts,
            f"{prefix} {model['key']}: missing or reordered contexts",
        )
        for context in model["contexts"]:
            require(
                context["candidate_batch_sizes"]
                == sorted(set(context["candidate_batch_sizes"])),
                f"{prefix} {model['key']}: invalid candidate batches",
            )
            for batch in context["batches"]:
                label = (
                    f"{prefix} {model['key']} "
                    f"S={context['sequence_length']} "
                    f"B={batch['batch_size']}:"
                )
                if schema_version >= 2:
                    validate_wall_timestamps(batch, label)
                if batch["status"] == "oom":
                    oom_batches += 1
                    continue
                require(batch["status"] == "complete", f"{label} unknown status")
                complete_batches += 1
                require(batch["stable"] is True, f"{label} warmup did not stabilize")
                require(
                    len(batch["measured_times_ms"]) == measured_steps,
                    f"{label} wrong measured timing count",
                )
                require(
                    len(batch["measured_losses"]) == measured_steps,
                    f"{label} wrong measured loss count",
                )
                if schema_version >= 2:
                    validate_step_timestamps(
                        batch,
                        "warmup",
                        batch["warmup_steps_executed"],
                        label,
                    )
                    validate_step_timestamps(
                        batch,
                        "measured",
                        measured_steps,
                        label,
                    )
                require(
                    all(
                        math.isfinite(value) and value > 0
                        for value in batch["measured_times_ms"]
                    ),
                    f"{label} invalid timing",
                )
                require(
                    math.isfinite(batch["tokens_per_second"])
                    and batch["tokens_per_second"] > 0,
                    f"{label} invalid throughput",
                )
                # A rented/overclocked board can exceed the vendor's nominal
                # reference-clock peak. Preserve and flag that raw value; a
                # separate configured-clock normalization handles comparison.
                require(
                    math.isfinite(batch["model_flops_utilization_pct"])
                    and batch["model_flops_utilization_pct"] > 0,
                    f"{label} invalid nominal-reference MFU",
                )
                require(
                    0 < batch["peak_allocated_bytes"] <= total_device_bytes,
                    f"{label} invalid allocated VRAM",
                )
                require(
                    0 < batch["peak_reserved_bytes"] <= total_device_bytes,
                    f"{label} invalid reserved VRAM",
                )
                require(
                    batch["selected_sdpa_backend"] is not None,
                    f"{label} SDPA backend was not captured",
                )

    require(
        complete_batches > 0 or (allow_all_oom and oom_batches > 0),
        f"{prefix} no successful batches",
    )
    return {
        "record": record,
        "complete_batches": complete_batches,
        "oom_batches": oom_batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--allow-all-oom",
        action="store_true",
        help="Accept a terminal frontier whose tested batches all OOM.",
    )
    args = parser.parse_args()

    validated = [
        validate(path, allow_all_oom=args.allow_all_oom) for path in args.inputs
    ]
    first = validated[0]["record"]
    reference_points = point_signature(first)
    reference_candidates = candidate_signature(first)
    seeds: set[int] = set()
    for item, path in zip(validated, args.inputs):
        record = item["record"]
        require(record["seed"] not in seeds, f"{path}: duplicate seed")
        seeds.add(record["seed"])
        require(
            record["environment"]["device_name"] == first["environment"]["device_name"],
            f"{path}: GPU model differs across replicates",
        )
        require(
            record["environment"]["nvidia_smi"].split(",")[1].strip()
            == first["environment"]["nvidia_smi"].split(",")[1].strip(),
            f"{path}: GPU UUID differs across replicates",
        )
        require(
            record["environment"]["dense_bf16_peak_tflops"]
            == first["environment"]["dense_bf16_peak_tflops"],
            f"{path}: MFU denominator differs across replicates",
        )
        require(
            record["protocol"] == first["protocol"],
            f"{path}: benchmark protocol differs across replicates",
        )
        require(
            candidate_signature(record) == reference_candidates,
            f"{path}: candidate grid differs across replicates",
        )
        require(
            point_signature(record) == reference_points,
            f"{path}: success/OOM frontier differs across replicates",
        )

    for path, item in zip(args.inputs, validated):
        record = item["record"]
        print(
            f"VALID {path.name}: seed={record['seed']}, "
            f"successful_batches={item['complete_batches']}, "
            f"oom_batches={item['oom_batches']}, "
            f"wall={record['suite_wall_seconds']:.2f}s"
        )
    print(
        f"VALID replicate set: gpu={first['environment']['device_name']}, "
        f"seeds={sorted(seeds)}, "
        f"rows={len(reference_points) * len(validated)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
