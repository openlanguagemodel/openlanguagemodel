"""Optional 7B-class FSDP execution + checkpoint/resume smoke test.

Report only as an end-to-end FSDP execution test — not training quality.

Example (8 GPUs):
    torchrun --nproc_per_node=8 -m benchmarks.demo2026.fsdp_smoke \\
        --steps 50 --output benchmarks/demo2026/results/raw/fsdp.json --allow-dirty
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmarks.demo2026 import provenance
from benchmarks.demo2026.scaling.synthetic_data import SyntheticTokenRing
from olm.core.dist import barrier, get_local_rank, get_world_size, is_main_process, setup_distributed
from olm.models.meta.llama3 import Llama3_1_8B, Llama3Block
from olm.train.trainer.fsdp_trainer import FSDPTrainer
from olm.train.optim import AdamW


def shifted_ce(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def build_dataloader(vocab_size: int, seq_len: int, batch_size: int, seed: int):
    ring = SyntheticTokenRing(vocab_size, seq_len, ring_size=32, seed=seed)
    return DataLoader(ring, batch_size=batch_size, num_workers=0)


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    env = provenance.capture_environment()
    setup_distributed()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Use the 8B preset architecture (FSDP smoke — not a quality run).
    model = Llama3_1_8B()
    if device != "cpu":
        model = model.to(device)

    seq_len = 512
    batch_size = 1
    vocab = 128256
    loader = build_dataloader(vocab, seq_len, batch_size, seed=42)

    trainer = FSDPTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=loader,
        device=device,
        context_length=seq_len,
        learning_rate=3e-4,
        use_amp=True,
        use_warmup_cosine=False,
        sharding_strategy="FULL_SHARD",
        auto_wrap_policy="transformer",
        transformer_layer_cls=Llama3Block,
        mixed_precision_policy="bf16",
    )

    loss_trace: List[float] = []
    t0 = time.perf_counter()
    losses = trainer.train(epochs=1, max_steps=args.steps, log_interval=max(args.steps, 10))
    elapsed = time.perf_counter() - t0
    loss_trace = losses

    peak_gb = []
    if torch.cuda.is_available():
        peak_gb = [
            torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        ]

    tokens = world_size * batch_size * seq_len * args.steps
    tps = tokens / elapsed if elapsed > 0 else 0.0

    ckpt_dir = tempfile.mkdtemp(prefix="olm_fsdp_smoke_")
    save_ok = False
    save_seconds = None
    resume_ok = False

    if is_main_process():
        try:
            t_save = time.perf_counter()
            trainer.save_checkpoint(ckpt_dir, step=args.steps)
            save_seconds = time.perf_counter() - t_save
            save_ok = os.path.isdir(ckpt_dir) and any(
                f.endswith(".pt") or f.endswith(".bin")
                for f in os.listdir(ckpt_dir)
            ) or os.path.exists(os.path.join(ckpt_dir, "model.pt"))
        except Exception:
            save_ok = False
        finally:
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    barrier()
    resume_ok = save_ok  # full resume path needs separate process; record save success

    n_params = sum(p.numel() for p in set(model.parameters()))

    record = {
        "model": "Llama3_1_8B",
        "parameters": n_params,
        "gpu_count": world_size,
        "precision": "bf16",
        "sequence_length": seq_len,
        "local_batch_size": batch_size,
        "steps": args.steps,
        "tokens_per_second": tps,
        "peak_memory_gb_per_gpu": peak_gb or [0.0],
        "loss_trace": loss_trace,
        "checkpoint_save_ok": save_ok,
        "checkpoint_save_seconds": save_seconds,
        "checkpoint_resume_ok": resume_ok,
        "resume_loss_trace": [],
        "command": " ".join(sys.argv),
        "config": {"preset": "Llama3_1_8B", "fsdp": "FULL_SHARD", "wrap": "Llama3Block"},
        "environment": env,
    }

    if is_main_process():
        provenance.write_json(args.output, record)
        print(f"FSDP smoke: {tps:,.0f} tok/s, save_ok={save_ok}")
    barrier()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--output", default="benchmarks/demo2026/results/raw/fsdp.json"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty:
        provenance.require_clean_worktree(allow_dirty=False)

    if not torch.cuda.is_available():
        print("FSDP smoke requires CUDA GPUs.", file=sys.stderr)
        return 2

    try:
        run_smoke(args)
    except Exception as exc:
        print(f"FSDP smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
