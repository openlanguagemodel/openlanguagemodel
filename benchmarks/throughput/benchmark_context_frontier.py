#!/usr/bin/env python3
"""Benchmark the OLM batch-size/context-length training frontier on one GPU.

The validated 2,048-token saturation result supplies a best batch per model.
For each new context length, the script scales that batch inversely with
context length to preserve approximately the same tokens per step, then tests
the neighboring lower and higher powers of two. The existing 2,048-token
measurements are embedded by reference rather than recomputed.

The autocast dtype (``bf16`` or ``fp16``) is inherited from the baseline so a
frontier can never mix precisions with the batch sweep it extends.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from benchmark_training_mfu import (
    AMP_DTYPES,
    AdamW,
    MODEL_CONFIGS,
    build_model,
    is_oom,
    make_grad_scaler,
    nvidia_smi_metadata,
    optimizer_state_metadata,
    parse_models,
    registered_peak,
    run_batch,
    unique_parameter_count,
    unique_tensor_bytes,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(arguments: Iterable[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 2 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected comma-separated context lengths of at least two"
        )
    return parsed


def nearest_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    return 2 ** round(math.log2(value))


def candidate_batches(
    baseline_batch: int,
    baseline_context: int,
    context_length: int,
) -> tuple[int, list[int]]:
    center = nearest_power_of_two(baseline_batch * baseline_context / context_length)
    candidates = {center, center * 2}
    if center > 1:
        candidates.add(center // 2)
    return center, sorted(candidates)


def load_baseline(path: Path, seed: int | None) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("benchmark") != "olm_single_gpu_training_mfu":
        raise ValueError(f"{path}: wrong baseline benchmark type")
    if record.get("status") != "complete":
        raise ValueError(f"{path}: baseline is incomplete")
    if seed is not None and record.get("seed") != seed:
        raise ValueError(
            f"{path}: baseline seed {record.get('seed')} does not match {seed}"
        )
    if record.get("sequence_length") != 2_048:
        raise ValueError(f"{path}: expected a 2,048-token baseline")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate-baselines",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Complete 2,048-token replicate baselines used to choose one "
            "shared batch-size reference by median tokens/second. The "
            "seed-matched --baseline remains the source of embedded results."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--models",
        type=parse_models,
        default=list(MODEL_CONFIGS),
    )
    parser.add_argument(
        "--contexts",
        type=parse_csv_ints,
        default=[512, 1_024, 4_096, 8_192],
    )
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--max-warmup-steps", type=int, default=50)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--stability-window", type=int, default=5)
    parser.add_argument("--required-stable-checks", type=int, default=2)
    parser.add_argument("--max-relative-drift", type=float, default=0.02)
    parser.add_argument("--max-relative-jitter", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.max_warmup_steps < 2 * args.stability_window:
        parser.error("--max-warmup-steps must be at least twice --stability-window")
    if 2_048 in args.contexts:
        parser.error("2,048 is reused from --baseline and must not be re-run")

    baseline = load_baseline(args.baseline, args.seed)
    candidate_baseline_paths = args.candidate_baselines or [args.baseline]
    candidate_baselines = [
        load_baseline(path, None) for path in candidate_baseline_paths
    ]
    baseline_uuid = baseline["environment"]["nvidia_smi"].split(",")[1].strip()
    for path, candidate_baseline in zip(candidate_baseline_paths, candidate_baselines):
        if (
            candidate_baseline["environment"]["device_name"]
            != baseline["environment"]["device_name"]
        ):
            parser.error(f"{path}: candidate-reference GPU model differs")
        if (
            candidate_baseline["environment"]["nvidia_smi"].split(",")[1].strip()
            != baseline_uuid
        ):
            parser.error(f"{path}: candidate-reference GPU UUID differs")
        if (
            candidate_baseline["environment"]["dense_bf16_peak_tflops"]
            != baseline["environment"]["dense_bf16_peak_tflops"]
        ):
            parser.error(f"{path}: candidate-reference MFU denominator differs")
        if candidate_baseline["protocol"] != baseline["protocol"]:
            parser.error(f"{path}: candidate-reference protocol differs")
        if candidate_baseline.get("amp_dtype", "bf16") != baseline.get(
            "amp_dtype", "bf16"
        ):
            parser.error(f"{path}: candidate-reference autocast dtype differs")

    candidate_model_maps = [
        {model["key"]: model for model in item["models"]}
        for item in candidate_baselines
    ]
    reference_best_batches: dict[str, int | None] = {}
    reference_batch_median_tps: dict[str, dict[int, float]] = {}
    for model_key in args.models:
        if any(model_key not in models for models in candidate_model_maps):
            parser.error(f"candidate references do not all contain model {model_key!r}")
        complete_by_reference = []
        for models in candidate_model_maps:
            complete_by_reference.append(
                {
                    batch["batch_size"]: batch["tokens_per_second"]
                    for batch in models[model_key]["batches"]
                    if batch["status"] == "complete"
                }
            )
        common_batches = set(complete_by_reference[0])
        for complete in complete_by_reference[1:]:
            common_batches.intersection_update(complete)
        medians = {
            batch: statistics.median(
                complete[batch] for complete in complete_by_reference
            )
            for batch in sorted(common_batches)
        }
        reference_batch_median_tps[model_key] = medians
        reference_best_batches[model_key] = (
            max(medians, key=lambda batch: (medians[batch], batch)) if medians else None
        )

    # Schema-v2 baselines predate --amp-dtype and are all BF16.
    amp_dtype_name = baseline.get("amp_dtype", "bf16")
    amp_dtype = AMP_DTYPES[amp_dtype_name]
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    peak_tflops = args.peak_tflops or registered_peak(device_name, amp_dtype_name)[0]
    if peak_tflops is None:
        parser.error(
            f"no dense {amp_dtype_name.upper()} peak is registered for "
            f"{device_name!r}; pass --peak-tflops"
        )
    if baseline["environment"]["device_name"] != device_name:
        parser.error("baseline GPU model does not match the current GPU")
    if baseline["environment"]["dense_bf16_peak_tflops"] != peak_tflops:
        parser.error("baseline and context sweep use different MFU denominators")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    baseline_models = {model["key"]: model for model in baseline["models"]}
    eligible_keys = [
        key
        for key in args.models
        if (
            baseline_models[key].get("best_batch_size") is not None
            and reference_best_batches[key] is not None
        )
    ]
    excluded = [
        {
            "key": key,
            "reason": "no successful batch at the 2,048-token baseline",
        }
        for key in args.models
        if key not in eligible_keys
    ]

    suite_start = time.perf_counter()
    total_device_bytes = torch.cuda.get_device_properties(0).total_memory
    record: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "olm_single_gpu_context_batch_frontier",
        "amp_dtype": amp_dtype_name,
        "status": "in_progress",
        "started_at": utc_now(),
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "working_directory": os.getcwd(),
        "process_id": os.getpid(),
        "seed": args.seed,
        "baseline": {
            "source": str(args.baseline),
            "sequence_length": baseline["sequence_length"],
            "reuse_note": (
                "The complete 2,048-token batch sweep is referenced, not rerun."
            ),
        },
        "context_lengths_run": args.contexts,
        "candidate_policy": (
            "For each model, select the batch with the highest median "
            "tokens/second across the candidate-reference 2,048-token "
            "replicates. Scale that shared batch inversely with context "
            "length, round to a power of two, and test the adjacent "
            "lower/upper powers."
        ),
        "candidate_batch_reference": {
            "sources": [str(path) for path in candidate_baseline_paths],
            "seeds": [item["seed"] for item in candidate_baselines],
            "aggregation": "median_tokens_per_second_by_common_batch",
            "best_batch_size_by_model": reference_best_batches,
            "median_tokens_per_second_by_model_and_batch": {
                key: {str(batch): value for batch, value in medians.items()}
                for key, medians in reference_batch_median_tps.items()
            },
        },
        "precision": baseline["precision"],
        "optimizer": baseline["optimizer"],
        "loss": baseline["loss"],
        "gradient_accumulation_steps": 1,
        "activation_checkpointing": False,
        "attention_backend": "pytorch_sdpa_auto",
        "compile": False,
        "mfu_flop_model": baseline["mfu_flop_model"],
        "protocol": {
            "max_warmup_steps": args.max_warmup_steps,
            "measured_steps": args.measured_steps,
            "stability_window": args.stability_window,
            "required_stable_checks": args.required_stable_checks,
            "max_relative_drift": args.max_relative_drift,
            "max_relative_jitter": args.max_relative_jitter,
        },
        "environment": {
            "device_name": device_name,
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "total_device_bytes": total_device_bytes,
            "total_device_gb": total_device_bytes / 1e9,
            "dense_bf16_peak_tflops": peak_tflops,
            "dense_bf16_peak_source": baseline["environment"]["dense_bf16_peak_source"],
            "dense_bf16_peak_derivation": baseline["environment"].get(
                "dense_bf16_peak_derivation"
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "nvidia_smi": nvidia_smi_metadata(),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_status": git_value(["status", "--short"]),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "excluded_models": excluded,
        "models": [],
    }
    write_record(args.output, record)

    for model_key in eligible_keys:
        config = MODEL_CONFIGS[model_key]
        baseline_model = baseline_models[model_key]
        baseline_best_batch = baseline_model["best_batch_size"]
        reference_best_batch = reference_best_batches[model_key]
        assert reference_best_batch is not None
        model_record: dict[str, Any] = {
            "key": model_key,
            "config": asdict(config),
            "started_at": utc_now(),
            "wall_start_unix_ns": time.time_ns(),
            "actual_unique_parameters": baseline_model["actual_unique_parameters"],
            "baseline_best_batch_size": baseline_best_batch,
            "baseline_best_tokens_per_second": baseline_model["best_tokens_per_second"],
            "baseline_best_model_flops_utilization_pct": baseline_model[
                "best_model_flops_utilization_pct"
            ],
            "candidate_reference_best_batch_size": reference_best_batch,
            "candidate_reference_median_tokens_per_second": (
                reference_batch_median_tps[model_key][reference_best_batch]
            ),
            "baseline_result_reference": str(args.baseline),
            "contexts": [],
        }
        record["models"].append(model_record)
        write_record(args.output, record)

        for context_length in args.contexts:
            center_batch, batches = candidate_batches(
                reference_best_batch,
                baseline["sequence_length"],
                context_length,
            )
            context_wall_start = time.perf_counter()
            context_record: dict[str, Any] = {
                "sequence_length": context_length,
                "started_at": utc_now(),
                "wall_start_unix_ns": time.time_ns(),
                "baseline_tokens_per_step": (
                    reference_best_batch * baseline["sequence_length"]
                ),
                "center_batch_size": center_batch,
                "candidate_batch_sizes": batches,
                "status": "running",
                "batches": [],
            }
            model_record["contexts"].append(context_record)
            write_record(args.output, record)
            print(
                f"\n[{config.label} S={context_length}] "
                f"building FP32 model; candidates={batches}",
                flush=True,
            )

            model = None
            optimizer = None
            scaler = None
            build_started_at = utc_now()
            build_wall_start_unix_ns = time.time_ns()
            build_wall_start = time.perf_counter()
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = build_model(config, args.seed, context_length)
                torch.cuda.synchronize()
                parameter_count = unique_parameter_count(model)
                if parameter_count != model_record["actual_unique_parameters"]:
                    raise RuntimeError("parameter count changed relative to baseline")
                optimizer = AdamW(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    fused=True,
                )
                scaler = make_grad_scaler(amp_dtype_name)
                context_record.update(
                    {
                        "parameter_bytes": unique_tensor_bytes(model.parameters()),
                        "buffer_bytes": unique_tensor_bytes(model.buffers()),
                        "build_peak_allocated_gb": (
                            torch.cuda.max_memory_allocated() / 1e9
                        ),
                        "build_peak_reserved_gb": (
                            torch.cuda.max_memory_reserved() / 1e9
                        ),
                        "build_started_at": build_started_at,
                        "build_finished_at": utc_now(),
                        "build_wall_start_unix_ns": build_wall_start_unix_ns,
                        "build_wall_finish_unix_ns": time.time_ns(),
                        "build_wall_seconds": (time.perf_counter() - build_wall_start),
                    }
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not is_oom(error):
                    raise
                context_record.update(
                    {
                        "status": "oom_during_model_build",
                        "error": str(error),
                        "build_started_at": build_started_at,
                        "build_finished_at": utc_now(),
                        "build_wall_start_unix_ns": build_wall_start_unix_ns,
                        "build_wall_finish_unix_ns": time.time_ns(),
                        "build_wall_seconds": (time.perf_counter() - build_wall_start),
                        "finished_at": utc_now(),
                        "wall_finish_unix_ns": time.time_ns(),
                        "context_wall_seconds": (
                            time.perf_counter() - context_wall_start
                        ),
                    }
                )
                write_record(args.output, record)
                del optimizer, model, scaler
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[{config.label} S={context_length}] "
                    "OOM during model construction",
                    flush=True,
                )
                continue

            for batch_size in batches:
                print(
                    f"[{config.label} S={context_length}] batch={batch_size}",
                    flush=True,
                )
                batch_attempt_started_at = utc_now()
                batch_attempt_wall_start_unix_ns = time.time_ns()
                try:
                    assert model is not None
                    assert optimizer is not None
                    result = run_batch(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        parameter_count=parameter_count,
                        batch_size=batch_size,
                        sequence_length=context_length,
                        seed=args.seed,
                        peak_tflops=peak_tflops,
                        total_device_bytes=total_device_bytes,
                        max_warmup_steps=args.max_warmup_steps,
                        measured_steps=args.measured_steps,
                        stability_window=args.stability_window,
                        required_stable_checks=args.required_stable_checks,
                        max_relative_drift=args.max_relative_drift,
                        max_relative_jitter=args.max_relative_jitter,
                        amp_dtype=amp_dtype,
                        scaler=scaler,
                    )
                    context_record["batches"].append(result)
                    context_record["optimizer_state"] = optimizer_state_metadata(
                        optimizer
                    )
                    write_record(args.output, record)
                    print(
                        f"[{config.label} S={context_length}] "
                        f"batch={batch_size} "
                        f"tok/s={result['tokens_per_second']:.1f} "
                        f"MFU={result['model_flops_utilization_pct']:.1f}% "
                        f"VRAM={result['peak_allocated_gb']:.2f}GB "
                        f"SDPA={result['selected_sdpa_backend']}",
                        flush=True,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    if not is_oom(error):
                        raise
                    allocated = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    context_record["batches"].append(
                        {
                            "status": "oom",
                            "started_at": batch_attempt_started_at,
                            "finished_at": utc_now(),
                            "wall_start_unix_ns": (batch_attempt_wall_start_unix_ns),
                            "wall_finish_unix_ns": time.time_ns(),
                            "batch_size": batch_size,
                            "seed": args.seed,
                            "error": str(error),
                            "peak_allocated_bytes": allocated,
                            "peak_allocated_gb": allocated / 1e9,
                            "peak_allocated_vram_pct": (
                                100.0 * allocated / total_device_bytes
                            ),
                            "peak_reserved_bytes": reserved,
                            "peak_reserved_gb": reserved / 1e9,
                            "peak_reserved_vram_pct": (
                                100.0 * reserved / total_device_bytes
                            ),
                        }
                    )
                    context_record["sweep_stop_reason"] = "out_of_memory"
                    write_record(args.output, record)
                    print(
                        f"[{config.label} S={context_length}] "
                        f"batch={batch_size} OOM",
                        flush=True,
                    )
                    break

            successful = [
                batch
                for batch in context_record["batches"]
                if batch["status"] == "complete"
            ]
            if successful:
                best = max(
                    successful,
                    key=lambda batch: batch["tokens_per_second"],
                )
                context_record.update(
                    {
                        "status": "complete",
                        "best_batch_size": best["batch_size"],
                        "best_tokens_per_second": best["tokens_per_second"],
                        "best_model_flops_utilization_pct": best[
                            "model_flops_utilization_pct"
                        ],
                        "best_peak_allocated_gb": best["peak_allocated_gb"],
                    }
                )
            else:
                context_record["status"] = "no_train_step_fit"
            context_record["finished_at"] = utc_now()
            context_record["wall_finish_unix_ns"] = time.time_ns()
            context_record["context_wall_seconds"] = (
                time.perf_counter() - context_wall_start
            )
            write_record(args.output, record)
            del optimizer, model, scaler
            gc.collect()
            torch.cuda.empty_cache()
        model_record["finished_at"] = utc_now()
        model_record["wall_finish_unix_ns"] = time.time_ns()

    record["status"] = "complete"
    record["finished_at"] = utc_now()
    record["suite_wall_seconds"] = time.perf_counter() - suite_start
    write_record(args.output, record)
    print(
        f"\nCompleted context frontier in "
        f"{record['suite_wall_seconds']:.2f}s: {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
