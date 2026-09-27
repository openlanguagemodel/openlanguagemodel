#!/usr/bin/env python3
"""Benchmark steady-state OLM training MFU on one CUDA GPU.

Each timed step contains a BF16-autocast forward pass, shifted-token
cross-entropy, backward pass, and fused AdamW update. Parameters, gradients,
and optimizer states use PyTorch's standard FP32 training storage. Batch 1 is
the fixed-workload result; larger power-of-two batches form the saturation
sweep.

GPUs without BF16 Tensor Cores (compute capability below 8.0, such as the
Tesla T4) can run the same step with ``--amp-dtype fp16``. The step then adds
dynamic loss scaling through ``torch.amp.GradScaler`` and MFU uses the GPU's
dense FP16 peak. For schema compatibility the denominator is still stored in
the ``dense_bf16_peak_tflops`` field; ``amp_dtype`` records which peak it is.

Warmup timings are retained so cold-start behavior and convergence to steady
state remain inspectable. Results are checkpointed to JSON after every model
build and batch, including OOM boundaries.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from olm.models.meta.llama3 import Llama3Model
from olm.train.optim import AdamW


@dataclass(frozen=True)
class ModelConfig:
    label: str
    target_parameters: int
    vocab_size: int
    embed_dim: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int


# Dense, tied-embedding, Llama-3-style models. Dimensions are hardware-friendly
# multiples; actual unique parameter counts are measured and written out.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "150m": ModelConfig("150M", 150_000_000, 32_000, 768, 2_048, 20, 12, 4),
    "250m": ModelConfig("250M", 250_000_000, 32_000, 896, 2_432, 26, 14, 2),
    "350m": ModelConfig("350M", 350_000_000, 32_000, 1_024, 2_816, 28, 16, 4),
    "500m": ModelConfig("500M", 500_000_000, 32_000, 1_280, 3_456, 27, 20, 4),
    "700m": ModelConfig("700M", 700_000_000, 32_000, 1_536, 4_096, 27, 24, 4),
    "1b": ModelConfig("1B", 1_000_000_000, 32_000, 1_792, 4_864, 28, 28, 4),
    "2b": ModelConfig("2B", 2_000_000_000, 32_000, 2_560, 6_912, 28, 20, 4),
    "4b": ModelConfig("4B", 4_000_000_000, 32_000, 3_584, 9_728, 29, 28, 4),
    "8b": ModelConfig("8B", 8_000_000_000, 32_000, 4_096, 14_336, 36, 32, 8),
}


# Dense BF16 Tensor Core throughput with FP32 accumulation. Sparse throughput
# and marketing "AI TOPS" are intentionally not used as MFU denominators.
KNOWN_DENSE_BF16_TFLOPS = {
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA GeForce RTX 5090": 209.5,
    "NVIDIA RTX A6000": 154.8,
    "NVIDIA A100 80GB PCIe": 312.0,
    "NVIDIA H100 80GB HBM3": 989.5,
    "NVIDIA H200": 989.5,
    "NVIDIA H200 NVL": 835.5,
    "NVIDIA B200": 2_250.0,
    "NVIDIA B300 SXM6 AC": 2_250.0,
    "NVIDIA B300-SXM6-AC": 2_250.0,
    "NVIDIA RTX 6000 Ada Generation": 364.2,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 468.0,
}

DENSE_BF16_PEAK_SOURCES = {
    "NVIDIA GeForce RTX 4090": (
        "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/"
        "nvidia-ada-gpu-architecture.pdf"
    ),
    "NVIDIA GeForce RTX 5090": (
        "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
        "nvidia-rtx-blackwell-gpu-architecture.pdf"
    ),
    "NVIDIA RTX A6000": (
        "https://www.nvidia.com/content/PDF/"
        "nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf"
    ),
    "NVIDIA A100 80GB PCIe": ("https://www.nvidia.com/en-us/data-center/a100/"),
    "NVIDIA H100 80GB HBM3": ("https://www.nvidia.com/en-us/data-center/h100/"),
    "NVIDIA H200": "https://www.nvidia.com/en-us/data-center/h200/",
    "NVIDIA H200 NVL": "https://www.nvidia.com/en-us/data-center/h200/",
    "NVIDIA B200": "https://www.nvidia.com/en-us/data-center/hgx/",
    "NVIDIA B300 SXM6 AC": "https://www.nvidia.com/en-us/data-center/hgx/",
    "NVIDIA B300-SXM6-AC": "https://www.nvidia.com/en-us/data-center/hgx/",
    "NVIDIA RTX 6000 Ada Generation": (
        "https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/"
        "NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf"
    ),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (
        "https://www.nvidia.com/en-us/data-center/"
        "rtx-pro-6000-blackwell-server-edition/"
    ),
}

DENSE_BF16_PEAK_DERIVATIONS = {
    "NVIDIA GeForce RTX 4090": (
        "Official dense BF16 Tensor TFLOP/s with FP32 accumulation."
    ),
    "NVIDIA GeForce RTX 5090": (
        "Official dense BF16 Tensor TFLOP/s with FP32 accumulation."
    ),
    "NVIDIA RTX A6000": (
        "NVIDIA specifies 154.8 TFLOP/s dense BF16 Tensor Core performance "
        "with FP32 accumulation at the 1,800 MHz reference boost clock. The "
        "separately listed 309.6 TFLOP/s value uses structured sparsity."
    ),
    "NVIDIA A100 80GB PCIe": (
        "NVIDIA specifies 312 TFLOP/s BF16 Tensor Core performance without "
        "sparsity. The separately listed 624 TFLOP/s value uses sparsity and "
        "is not an MFU denominator."
    ),
    "NVIDIA H100 80GB HBM3": (
        "Half NVIDIA's official 1,979 TFLOP/s BF16 Tensor Core "
        "specification because the published value uses sparsity. The dense "
        "MFU denominator is therefore 989.5 TFLOP/s."
    ),
    "NVIDIA H200": (
        "Half NVIDIA's official 1,979 TFLOP/s H200 SXM BF16 Tensor Core "
        "specification because the published value uses sparsity. The dense "
        "MFU denominator is therefore 989.5 TFLOP/s."
    ),
    "NVIDIA H200 NVL": (
        "Half NVIDIA's official 1,671 TFLOP/s H200 NVL BF16 Tensor Core "
        "specification because the published value uses sparsity. The dense "
        "MFU denominator is therefore 835.5 TFLOP/s."
    ),
    "NVIDIA B200": (
        "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core performance "
        "for eight HGX B200 GPUs. Dense throughput is one-half of sparse, "
        "then divided by eight GPUs: 36,000 / 2 / 8 = 2,250 TFLOP/s."
    ),
    "NVIDIA B300 SXM6 AC": (
        "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core performance "
        "for eight HGX B300 GPUs. Dense throughput is one-half of sparse, "
        "then divided by eight GPUs: 36,000 / 2 / 8 = 2,250 TFLOP/s."
    ),
    "NVIDIA B300-SXM6-AC": (
        "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core performance "
        "for eight HGX B300 GPUs. Dense throughput is one-half of sparse, "
        "then divided by eight GPUs: 36,000 / 2 / 8 = 2,250 TFLOP/s."
    ),
    "NVIDIA RTX 6000 Ada Generation": (
        "NVIDIA's official architecture whitepaper specifies 364.2 TFLOP/s "
        "dense BF16 Tensor Core performance with FP32 accumulation. The "
        "separately listed 728.4 TFLOP/s value uses structured sparsity."
    ),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (
        "Twice NVIDIA's official 234 TFLOP/s dense TF32 Tensor Core "
        "specification. The separately advertised 1 PFLOP FP16/BF16 value "
        "uses sparsity and is not an MFU denominator."
    ),
}


# Dense FP16 Tensor Core throughput with FP32 accumulation. Used as the MFU
# denominator for ``--amp-dtype fp16`` runs on GPUs without BF16 Tensor Cores.
KNOWN_DENSE_FP16_TFLOPS = {
    "Tesla T4": 65.0,
}

DENSE_FP16_PEAK_SOURCES = {
    "Tesla T4": "https://www.nvidia.com/en-us/data-center/tesla-t4/",
}

DENSE_FP16_PEAK_DERIVATIONS = {
    "Tesla T4": (
        "NVIDIA specifies 65 TFLOP/s mixed-precision (FP16 inputs, FP32 "
        "accumulation) Tensor Core performance. Turing has no structured-"
        "sparsity mode, so the published value is already a dense rate."
    ),
}

AMP_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def registered_peak(
    device_name: str,
    amp_dtype: str,
) -> tuple[float | None, str | None, str | None]:
    """Return the registered dense peak, its source, and its derivation."""

    if amp_dtype == "fp16":
        return (
            KNOWN_DENSE_FP16_TFLOPS.get(device_name),
            DENSE_FP16_PEAK_SOURCES.get(device_name),
            DENSE_FP16_PEAK_DERIVATIONS.get(device_name),
        )
    return (
        KNOWN_DENSE_BF16_TFLOPS.get(device_name),
        DENSE_BF16_PEAK_SOURCES.get(device_name),
        DENSE_BF16_PEAK_DERIVATIONS.get(device_name),
    )


def make_grad_scaler(amp_dtype: str) -> torch.amp.GradScaler | None:
    """FP16 needs dynamic loss scaling; BF16 has FP32's exponent range."""

    if amp_dtype == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_parameter_count(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += parameter.numel()
    return total


def unique_tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        identity = id(tensor)
        if identity not in seen:
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
    return total


def optimizer_state_metadata(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return {
        "tensor_bytes": unique_tensor_bytes(tensors),
        "tensor_gb": unique_tensor_bytes(tensors) / 1e9,
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def robust_relative_jitter(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0:
        return math.inf
    mad = statistics.median(abs(value - median) for value in values)
    return 1.4826 * mad / median


def stability_metrics(times_ms: list[float], window: int) -> dict[str, float] | None:
    if len(times_ms) < 2 * window:
        return None
    previous = times_ms[-2 * window : -window]
    current = times_ms[-window:]
    previous_median = statistics.median(previous)
    current_median = statistics.median(current)
    drift = abs(current_median - previous_median) / max(previous_median, 1e-12)
    return {
        "previous_median_ms": previous_median,
        "current_median_ms": current_median,
        "relative_median_drift": drift,
        "robust_relative_jitter": robust_relative_jitter(current),
    }


def training_flops(
    parameter_count: int,
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
) -> int:
    """Standard dense-transformer training FLOP approximation.

    Per token this is ``6N + 12 * layers * hidden * sequence_length``:
    three times the customary forward FLOPs, including the causal-attention
    sequence-length correction. Elementwise operations are omitted.
    """

    flops_per_token = (
        6 * parameter_count
        + 12 * config.num_layers * config.embed_dim * sequence_length
    )
    return batch_size * sequence_length * flops_per_token


def shifted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def selected_sdpa_backend(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype = torch.bfloat16,
) -> str | None:
    """Ask PyTorch which SDPA backend the benchmark tensor shape selects."""

    try:
        head_dim = config.embed_dim // config.num_heads
        # The OLM GQA implementation expands K/V to the query-head count before
        # SDPA. One tensor is sufficient because the dispatcher only inspects
        # shape, dtype, strides, and device for this query.
        probe = torch.empty(
            (batch_size, config.num_heads, sequence_length, head_dim),
            device="cuda",
            dtype=dtype,
        )
        choice = torch._fused_sdp_choice(
            probe,
            probe,
            probe,
            None,
            0.0,
            True,
        )
        backend = SDPBackend(choice).name
        del probe
        return backend
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def timed_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    start_event.record()
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(tokens)
        loss = shifted_cross_entropy(logits, tokens)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        # Fused AdamW unscales and skips overflowing steps inside its update
        # kernel; GradScaler adds one non-finite check over the gradients.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    loss_value = float(loss.detach())
    del logits, loss
    return elapsed_ms, loss_value


def run_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    parameter_count: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
    peak_tflops: float,
    total_device_bytes: int,
    max_warmup_steps: int,
    measured_steps: int,
    stability_window: int,
    required_stable_checks: int,
    max_relative_drift: float,
    max_relative_jitter: float,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    batch_started_at = utc_now()
    batch_wall_start_unix_ns = time.time_ns()
    batch_wall_start = time.perf_counter()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device="cuda",
        dtype=torch.long,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    warmup_times_ms: list[float] = []
    warmup_losses: list[float] = []
    warmup_step_wall_start_unix_ns: list[int] = []
    warmup_step_wall_finish_unix_ns: list[int] = []
    stable_checks = 0
    stable = False
    final_stability: dict[str, float] | None = None

    for _ in range(max_warmup_steps):
        warmup_step_wall_start_unix_ns.append(time.time_ns())
        elapsed_ms, loss_value = timed_training_step(
            model, optimizer, tokens, start_event, end_event, amp_dtype, scaler
        )
        warmup_step_wall_finish_unix_ns.append(time.time_ns())
        warmup_times_ms.append(elapsed_ms)
        warmup_losses.append(loss_value)
        metrics = stability_metrics(warmup_times_ms, stability_window)
        if metrics is None:
            continue
        final_stability = metrics
        within_threshold = (
            metrics["relative_median_drift"] <= max_relative_drift
            and metrics["robust_relative_jitter"] <= max_relative_jitter
        )
        stable_checks = stable_checks + 1 if within_threshold else 0
        if stable_checks >= required_stable_checks:
            stable = True
            break

    measured_times_ms: list[float] = []
    measured_losses: list[float] = []
    measured_step_wall_start_unix_ns: list[int] = []
    measured_step_wall_finish_unix_ns: list[int] = []
    for _ in range(measured_steps):
        measured_step_wall_start_unix_ns.append(time.time_ns())
        elapsed_ms, loss_value = timed_training_step(
            model, optimizer, tokens, start_event, end_event, amp_dtype, scaler
        )
        measured_step_wall_finish_unix_ns.append(time.time_ns())
        measured_times_ms.append(elapsed_ms)
        measured_losses.append(loss_value)

    median_ms = statistics.median(measured_times_ms)
    mean_ms = statistics.mean(measured_times_ms)
    total_flops = training_flops(parameter_count, config, batch_size, sequence_length)
    achieved_tflops = total_flops / (median_ms / 1_000.0) / 1e12
    tokens_per_second = batch_size * sequence_length / (median_ms / 1_000.0)
    mean_tokens_per_second = batch_size * sequence_length / (mean_ms / 1_000.0)
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    sdpa_backend = selected_sdpa_backend(config, batch_size, sequence_length, amp_dtype)
    batch_finished_at = utc_now()
    batch_wall_finish_unix_ns = time.time_ns()

    result = {
        "status": "complete",
        "started_at": batch_started_at,
        "finished_at": batch_finished_at,
        "wall_start_unix_ns": batch_wall_start_unix_ns,
        "wall_finish_unix_ns": batch_wall_finish_unix_ns,
        "batch_size": batch_size,
        "seed": seed,
        "stable": stable,
        "warmup_steps_executed": len(warmup_times_ms),
        "warmup_times_ms": warmup_times_ms,
        "warmup_losses": warmup_losses,
        "warmup_step_wall_start_unix_ns": warmup_step_wall_start_unix_ns,
        "warmup_step_wall_finish_unix_ns": warmup_step_wall_finish_unix_ns,
        "final_stability_metrics": final_stability,
        "measured_steps": measured_steps,
        "measured_times_ms": measured_times_ms,
        "measured_losses": measured_losses,
        "measured_step_wall_start_unix_ns": measured_step_wall_start_unix_ns,
        "measured_step_wall_finish_unix_ns": measured_step_wall_finish_unix_ns,
        "mean_step_time_ms": mean_ms,
        "median_step_time_ms": median_ms,
        "p25_step_time_ms": percentile(measured_times_ms, 0.25),
        "p75_step_time_ms": percentile(measured_times_ms, 0.75),
        "measured_robust_relative_jitter": robust_relative_jitter(measured_times_ms),
        "tokens_per_second": tokens_per_second,
        "mean_based_tokens_per_second": mean_tokens_per_second,
        "tokens_per_step": batch_size * sequence_length,
        "loss_tokens_per_step": batch_size * (sequence_length - 1),
        "training_flops_per_step": total_flops,
        "achieved_tflops": achieved_tflops,
        "dense_bf16_peak_tflops": peak_tflops,
        "model_flops_utilization_pct": 100.0 * achieved_tflops / peak_tflops,
        "selected_sdpa_backend": sdpa_backend,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_gb": peak_allocated_bytes / 1e9,
        "peak_allocated_vram_pct": 100.0 * peak_allocated_bytes / total_device_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_gb": peak_reserved_bytes / 1e9,
        "peak_reserved_vram_pct": 100.0 * peak_reserved_bytes / total_device_bytes,
        "batch_wall_seconds": time.perf_counter() - batch_wall_start,
    }
    if scaler is not None:
        result["grad_scale"] = scaler.get_scale()
    optimizer.zero_grad(set_to_none=True)
    del tokens
    return result


def build_model(
    config: ModelConfig,
    seed: int,
    sequence_length: int,
) -> Llama3Model:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Construct standard FP32 parameters directly on the GPU to avoid a
    # transient host copy. BF16 is used for compute through autocast.
    with torch.device("cuda"):
        model = Llama3Model(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_seq_len=sequence_length,
            rope_theta=500_000.0,
            dropout=0.0,
            tie_weights=True,
        )
    return model.train()


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


def nvidia_smi_metadata() -> str | None:
    fields = (
        "name,uuid,pci.bus_id,driver_version,pstate,power.limit,"
        "clocks.max.sm,clocks.max.memory,memory.total"
    )
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
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
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_CONFIGS)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(MODEL_CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model labels: {', '.join(unknown)}")
    return selected


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models", type=parse_models, default=list(MODEL_CONFIGS))
    parser.add_argument(
        "--batches",
        type=parse_csv_ints,
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Power-of-two saturation sweep; batch 1 is the fixed workload.",
    )
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument(
        "--amp-dtype",
        choices=sorted(AMP_DTYPES),
        default="bf16",
        help=(
            "Autocast compute dtype. Use fp16 on GPUs without BF16 Tensor Cores "
            "(compute capability below 8.0, such as the Tesla T4)."
        ),
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
    parser.add_argument(
        "--stop-after-non-improving",
        type=int,
        default=2,
        help=(
            "Stop after this many consecutive batch doublings fail to improve "
            "tokens/second; pass 0 to exhaust every batch."
        ),
    )
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.01,
        help="Relative throughput gain required to reset the saturation counter.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2 for shifted-token loss")
    if args.measured_steps < 1:
        parser.error("--measured-steps must be positive")
    if args.max_warmup_steps < 2 * args.stability_window:
        parser.error("--max-warmup-steps must be at least twice --stability-window")
    if args.required_stable_checks < 1:
        parser.error("--required-stable-checks must be positive")

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    if args.amp_dtype == "bf16" and device_capability < (8, 0):
        parser.error(
            f"{device_name!r} (compute capability "
            f"{device_capability[0]}.{device_capability[1]}) has no BF16 Tensor "
            "Cores; pass --amp-dtype fp16"
        )
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    registered_tflops, peak_source, peak_derivation = registered_peak(
        device_name, args.amp_dtype
    )
    peak_tflops = args.peak_tflops or registered_tflops
    if peak_tflops is None:
        parser.error(
            f"no dense {args.amp_dtype.upper()} peak is registered for "
            f"{device_name!r}; pass --peak-tflops"
        )
    if args.peak_tflops is not None:
        peak_source = peak_derivation = "User-supplied --peak-tflops value."

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    suite_start = time.perf_counter()
    total_device_bytes = torch.cuda.get_device_properties(0).total_memory
    record: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "olm_single_gpu_training_mfu",
        "metric_note": (
            "Every timed step includes forward, shifted-token cross-entropy, "
            "backward, and fused AdamW update."
        ),
        "amp_dtype": args.amp_dtype,
        "status": "in_progress",
        "started_at": utc_now(),
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "working_directory": os.getcwd(),
        "process_id": os.getpid(),
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "batch_sizes": args.batches,
        "precision": {
            "parameter_dtype": "torch.float32",
            "compute_autocast_dtype": str(amp_dtype),
            "gradient_dtype": "torch.float32",
            "optimizer_state_dtype": "torch.float32",
            "loss_scaling": (
                "torch.amp.GradScaler (dynamic)" if args.amp_dtype == "fp16" else None
            ),
        },
        "optimizer": {
            "implementation": "olm.train.optim.AdamW",
            "fused": True,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "loss": "shifted_token_cross_entropy",
        "gradient_accumulation_steps": 1,
        "activation_checkpointing": False,
        "attention_backend": "pytorch_sdpa_auto",
        "compile": False,
        "mfu_flop_model": {
            "formula_per_token": "6*N + 12*L*H*S",
            "notes": (
                "Standard dense-transformer approximation; elementwise "
                "operations are omitted."
            ),
        },
        "protocol": {
            "max_warmup_steps": args.max_warmup_steps,
            "measured_steps": args.measured_steps,
            "stability_window": args.stability_window,
            "required_stable_checks": args.required_stable_checks,
            "max_relative_drift": args.max_relative_drift,
            "max_relative_jitter": args.max_relative_jitter,
            "stop_after_non_improving": args.stop_after_non_improving,
            "minimum_relative_improvement": args.minimum_relative_improvement,
        },
        "environment": {
            "device_name": device_name,
            "device_capability": list(device_capability),
            "total_device_bytes": total_device_bytes,
            "total_device_gb": total_device_bytes / 1e9,
            # The peak for ``amp_dtype``; the field name predates FP16 support.
            "dense_bf16_peak_tflops": peak_tflops,
            "dense_bf16_peak_source": peak_source,
            "dense_bf16_peak_derivation": peak_derivation,
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
        "models": [],
    }
    write_record(args.output, record)

    for model_key in args.models:
        config = MODEL_CONFIGS[model_key]
        print(f"\n[{config.label}] building FP32 model", flush=True)
        model_started_at = utc_now()
        model_wall_start_unix_ns = time.time_ns()
        model_wall_start = time.perf_counter()
        model: Llama3Model | None = None
        optimizer: torch.optim.Optimizer | None = None
        scaler: torch.amp.GradScaler | None = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(config, args.seed, args.sequence_length)
            torch.cuda.synchronize()
            parameter_count = unique_parameter_count(model)
            build_seconds = time.perf_counter() - model_wall_start
            optimizer = AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                fused=True,
            )
            scaler = make_grad_scaler(args.amp_dtype)
            parameter_bytes = unique_tensor_bytes(model.parameters())
            buffer_bytes = unique_tensor_bytes(model.buffers())
            model_record: dict[str, Any] = {
                "key": model_key,
                "config": asdict(config),
                "status": "running",
                "started_at": model_started_at,
                "wall_start_unix_ns": model_wall_start_unix_ns,
                "actual_unique_parameters": parameter_count,
                "target_error_pct": 100.0
                * (parameter_count - config.target_parameters)
                / config.target_parameters,
                "parameter_bytes": parameter_bytes,
                "parameter_gb": parameter_bytes / 1e9,
                "buffer_bytes": buffer_bytes,
                "buffer_gb": buffer_bytes / 1e9,
                "build_peak_allocated_gb": (torch.cuda.max_memory_allocated() / 1e9),
                "build_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "build_seconds": build_seconds,
                "batches": [],
            }
            record["models"].append(model_record)
            write_record(args.output, record)
            print(
                f"[{config.label}] {parameter_count:,} params; "
                f"build={build_seconds:.2f}s; params={parameter_bytes / 1e9:.2f}GB",
                flush=True,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not is_oom(error):
                raise
            model_record = {
                "key": model_key,
                "config": asdict(config),
                "status": "oom_during_model_build",
                "started_at": model_started_at,
                "finished_at": utc_now(),
                "wall_start_unix_ns": model_wall_start_unix_ns,
                "wall_finish_unix_ns": time.time_ns(),
                "error": str(error),
                "build_peak_allocated_gb": (torch.cuda.max_memory_allocated() / 1e9),
                "build_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "build_seconds": time.perf_counter() - model_wall_start,
                "batches": [],
            }
            record["models"].append(model_record)
            write_record(args.output, record)
            del optimizer, model, scaler
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[{config.label}] OOM during model construction", flush=True)
            continue

        saturation_reference = -math.inf
        non_improving_batches = 0
        for batch_size in args.batches:
            print(f"[{config.label}] batch={batch_size}", flush=True)
            batch_attempt_started_at = utc_now()
            batch_attempt_wall_start_unix_ns = time.time_ns()
            try:
                assert model is not None
                assert optimizer is not None
                batch_result = run_batch(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    parameter_count=parameter_count,
                    batch_size=batch_size,
                    sequence_length=args.sequence_length,
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
                model_record["batches"].append(batch_result)
                model_record["optimizer_state"] = optimizer_state_metadata(optimizer)
                write_record(args.output, record)
                print(
                    f"[{config.label}] batch={batch_size} "
                    f"median={batch_result['median_step_time_ms']:.2f}ms "
                    f"tok/s={batch_result['tokens_per_second']:.1f} "
                    f"MFU={batch_result['model_flops_utilization_pct']:.1f}% "
                    f"warmup={batch_result['warmup_steps_executed']} "
                    f"VRAM={batch_result['peak_allocated_gb']:.2f}GB",
                    flush=True,
                )
                throughput = batch_result["tokens_per_second"]
                required_throughput = saturation_reference * (
                    1.0 + args.minimum_relative_improvement
                )
                if throughput >= required_throughput:
                    saturation_reference = throughput
                    non_improving_batches = 0
                else:
                    non_improving_batches += 1
                if (
                    args.stop_after_non_improving > 0
                    and non_improving_batches >= args.stop_after_non_improving
                ):
                    model_record["sweep_stop_reason"] = (
                        "saturation: consecutive batch doublings did not improve "
                        f"throughput by {args.minimum_relative_improvement:.1%}"
                    )
                    print(
                        f"[{config.label}] saturation established; stopping sweep",
                        flush=True,
                    )
                    break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not is_oom(error):
                    raise
                oom_allocated = torch.cuda.max_memory_allocated()
                oom_reserved = torch.cuda.max_memory_reserved()
                model_record["batches"].append(
                    {
                        "status": "oom",
                        "started_at": batch_attempt_started_at,
                        "finished_at": utc_now(),
                        "wall_start_unix_ns": batch_attempt_wall_start_unix_ns,
                        "wall_finish_unix_ns": time.time_ns(),
                        "batch_size": batch_size,
                        "seed": args.seed,
                        "error": str(error),
                        "peak_allocated_bytes": oom_allocated,
                        "peak_allocated_gb": oom_allocated / 1e9,
                        "peak_allocated_vram_pct": 100.0
                        * oom_allocated
                        / total_device_bytes,
                        "peak_reserved_bytes": oom_reserved,
                        "peak_reserved_gb": oom_reserved / 1e9,
                        "peak_reserved_vram_pct": 100.0
                        * oom_reserved
                        / total_device_bytes,
                    }
                )
                model_record["sweep_stop_reason"] = "out_of_memory"
                write_record(args.output, record)
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[{config.label}] batch={batch_size} OOM; stopping sweep",
                    flush=True,
                )
                break

        complete_batches = [
            batch
            for batch in model_record["batches"]
            if batch.get("status") == "complete"
        ]
        if complete_batches:
            best = max(
                complete_batches,
                key=lambda batch: batch["tokens_per_second"],
            )
            model_record["best_batch_size"] = best["batch_size"]
            model_record["best_tokens_per_second"] = best["tokens_per_second"]
            model_record["best_model_flops_utilization_pct"] = best[
                "model_flops_utilization_pct"
            ]
            model_record["best_peak_allocated_gb"] = best["peak_allocated_gb"]
            model_record["status"] = "complete"
        else:
            model_record["status"] = "no_train_step_fit"
        model_record["finished_at"] = utc_now()
        model_record["wall_finish_unix_ns"] = time.time_ns()
        model_record["model_wall_seconds"] = time.perf_counter() - model_wall_start
        write_record(args.output, record)
        del optimizer, model, scaler
        gc.collect()
        torch.cuda.empty_cache()

    record["status"] = "complete"
    record["finished_at"] = utc_now()
    record["suite_wall_seconds"] = time.perf_counter() - suite_start
    write_record(args.output, record)
    print(f"\nCompleted in {record['suite_wall_seconds']:.2f}s: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
