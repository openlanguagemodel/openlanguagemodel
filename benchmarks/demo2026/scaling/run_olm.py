"""Measurement-grade OLM throughput / weak-scaling harness.

Uses a dedicated training loop with CUDA event timing (not the OLM Trainer's
``time.time()`` path). Supports single-GPU and DDP via ``torchrun``.

Example dry-run (CPU or 1 GPU, short):
    python -m benchmarks.demo2026.scaling.run_olm \\
        --config benchmarks/demo2026/configs/scaling/llama400m.yaml \\
        --gpu-count 1 --warmup-steps 2 --measured-steps 3 --allow-dirty

Full weak-scaling matrix:
    bash benchmarks/demo2026/scaling/run_scaling.sh
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from benchmarks.demo2026 import provenance
from benchmarks.demo2026.scaling.synthetic_data import SyntheticTokenRing
from olm.core.dist import barrier, get_local_rank, get_world_size, is_main_process, setup_distributed
from olm.models.meta.llama3 import Llama3Model


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def count_unique_params(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def build_model(cfg: Dict[str, Any], device: torch.device) -> Llama3Model:
    m = cfg["model"]
    model = Llama3Model(
        vocab_size=m["vocab_size"],
        embed_dim=m["embed_dim"],
        intermediate_size=m["intermediate_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        num_kv_heads=m["num_kv_heads"],
        max_seq_len=m["max_seq_len"],
        rope_theta=m["rope_theta"],
        dropout=m.get("dropout", 0.0),
        tie_weights=m.get("tie_weights", True),
    )
    return model.to(device)


def configure_sdpa(backend: str) -> None:
    if not torch.cuda.is_available():
        return
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        # sdpa / flash: let PyTorch pick (Flash when available)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    warmup_steps: int,
    measured_steps: int,
    grad_accum: int,
) -> Dict[str, Any]:
    use_cuda = device.type == "cuda"
    use_bf16 = precision == "bf16" and use_cuda
    use_fp16 = precision == "fp16" and use_cuda

    model.train()
    step_times_ms: List[float] = []
    loss_trace: List[float] = []
    data_iter = iter(loader)

    def next_batch():
        x, y = next(data_iter)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    total_steps = warmup_steps + measured_steps
    accum_loss = 0.0
    accum_count = 0

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = torch.cuda.Event(enable_timing=True) if use_cuda else None
        step_end = torch.cuda.Event(enable_timing=True) if use_cuda else None
        cpu_t0 = time.perf_counter() if not use_cuda else None

        if use_cuda:
            step_start.record()

        for micro in range(grad_accum):
            x, y = next_batch()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32,
                enabled=use_bf16 or use_fp16,
            ):
                logits = model(x)
                loss = shifted_ce(logits, y) / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum
            accum_count += 1

        optimizer.step()

        if use_cuda:
            step_end.record()
            torch.cuda.synchronize()
            elapsed_ms = step_start.elapsed_time(step_end)
        else:
            elapsed_ms = (time.perf_counter() - cpu_t0) * 1000.0

        if step >= warmup_steps:
            step_times_ms.append(elapsed_ms)
            loss_trace.append(accum_loss / max(accum_count, 1))
        accum_loss = 0.0
        accum_count = 0

    return {
        "step_times_ms": step_times_ms,
        "loss_trace": loss_trace,
    }


def peak_memory_gb() -> List[float]:
    """Report peak allocated memory on this rank's device only."""
    if not torch.cuda.is_available():
        return [0.0]
    local = get_local_rank()
    return [torch.cuda.max_memory_allocated(local) / 1e9]


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    env = provenance.capture_environment()
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    requested_gpus = args.gpu_count
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Bind CUDA device BEFORE process-group init so every rank stays on its GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    setup_distributed()
    world_size = get_world_size()

    if world_size != requested_gpus:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} does not match --gpu-count={requested_gpus}"
        )

    configure_sdpa(train_cfg.get("attention_backend", "sdpa"))

    model = build_model(cfg, device)
    n_params = count_unique_params(model)
    if is_main_process():
        print(f"Model parameters (unique): {n_params:,}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )

    compile_flag = train_cfg.get("compile", False)
    if compile_flag:
        model = torch.compile(model)

    seq_len = train_cfg["sequence_length"]
    local_batch = train_cfg["local_batch_size"]
    ring = SyntheticTokenRing(
        model_cfg["vocab_size"],
        seq_len,
        train_cfg.get("synthetic_ring_size", 64),
        train_cfg.get("seed", 42) + args.replicate,
    )
    loader = DataLoader(ring, batch_size=local_batch, num_workers=0)

    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 0.0)
    fused = train_cfg.get("fused_optimizer", False)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, fused=fused and device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup = args.warmup_steps if args.warmup_steps is not None else train_cfg["warmup_steps"]
    measured = args.measured_steps if args.measured_steps is not None else train_cfg["measured_steps"]
    grad_accum = train_cfg.get("grad_accum_steps", 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    barrier()

    metrics = train_loop(
        model,
        optimizer,
        loader,
        device,
        train_cfg.get("precision", "bf16"),
        warmup,
        measured,
        grad_accum,
    )
    barrier()

    mean_ms = statistics.mean(metrics["step_times_ms"]) if metrics["step_times_ms"] else 0.0
    std_ms = (
        statistics.stdev(metrics["step_times_ms"])
        if len(metrics["step_times_ms"]) > 1
        else 0.0
    )
    tokens_per_step = world_size * local_batch * seq_len
    tokens_per_sec = tokens_per_step / (mean_ms / 1000.0) if mean_ms > 0 else 0.0

    gpu_model = (
        torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu"
    )

    record = {
        "framework": cfg.get("framework", "olm"),
        "framework_version": env.get("olm_version"),
        "model": cfg.get("name", "llama400m"),
        "parameters": n_params,
        "gpu_model": gpu_model,
        "gpu_count": world_size,
        "interconnect": cfg.get("interconnect", "unknown"),
        "precision": train_cfg.get("precision", "bf16"),
        "sequence_length": seq_len,
        "local_batch_size": local_batch,
        "global_batch_size": world_size * local_batch * grad_accum,
        "grad_accum_steps": grad_accum,
        "optimizer": train_cfg.get("optimizer", "adamw"),
        "attention_backend": train_cfg.get("attention_backend", "sdpa"),
        "compile": compile_flag,
        "warmup_steps": warmup,
        "measured_steps": measured,
        "replicate": args.replicate,
        "seed": train_cfg.get("seed", 42) + args.replicate,
        "mean_step_time_ms": mean_ms,
        "step_time_std_ms": std_ms,
        "tokens_per_second": tokens_per_sec,
        "peak_memory_gb_per_gpu": peak_memory_gb(),
        "loss_first": metrics["loss_trace"][0] if metrics["loss_trace"] else None,
        "loss_last": metrics["loss_trace"][-1] if metrics["loss_trace"] else None,
        "loss_trace": metrics["loss_trace"],
        "config": cfg,
        "config_hash": provenance.config_hash(cfg),
        "command": " ".join(sys.argv),
        "reportable": not env.get("git_dirty", True) and not args.allow_dirty,
        "environment": env,
    }

    if is_main_process():
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(
            args.output,
            f"olm_{world_size}gpu_rep{args.replicate}.json",
        )
        provenance.write_json(out_path, record)
        print(
            f"[{world_size} GPU rep={args.replicate}] "
            f"{tokens_per_sec:,.0f} tok/s  mean_step={mean_ms:.2f} ms  "
            f"params={n_params:,}"
        )
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--measured-steps", type=int, default=None)
    parser.add_argument(
        "--output", default="benchmarks/demo2026/results/raw/scaling"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty:
        provenance.require_clean_worktree(allow_dirty=False)

    try:
        run_benchmark(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
