#!/usr/bin/env python3
"""
GPT-2 124M Training on FineWeb Edu 10B Tokens

Usage:
    python -u train.py --config config.yaml
    python -u train.py --config config.yaml --resume checkpoints/step_5000.pt

    Note: Use -u flag for unbuffered output when redirecting to log files:
    nohup python -u train.py --config config.yaml > logs/train.log 2>&1 &
    nohup python -u train.py --config config.yaml  --resume checkpoints/step_9000.pt > logs/train.log 2>&1 &
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from olm.models.openai import GPT2
from olm.data.datasets.fineweb_edu import FineWebEduDataset
from olm.train.optim import AdamW
from olm.data.datasets import DataLoader
from olm.train.schedulers.cosine import CosineAnnealingLR
from olm.train.schedulers.warmup import WarmupLR
from olm.train.trainer import (
    Trainer,
    ValidationCallback,
    CheckpointCallback,
    MetricsLoggerCallback,
    ThroughputCallback,
)


def setup_training(config: Dict[str, Any], resume_path: Optional[str] = None):
    """Setup training components."""

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Calculate skip_batches if resuming
    skip_batches = 0
    resume_step = 0
    if resume_path:
        # Load checkpoint to get the step number
        checkpoint = torch.load(resume_path, map_location="cpu")
        resume_step = checkpoint.get("step", 0)
        # Each step processes batch_size samples (gradient accumulation is internal to training loop)
        train_config = config["training"]
        skip_batches = resume_step * train_config["batch_size"]
        del checkpoint  # Free memory
        print(
            f"Will skip {skip_batches} batches (dataset samples) to resume from step {resume_step}"
        )

    # Model
    print("Initializing GPT-2 model...")
    model = GPT2().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Data
    print("Loading FineWeb Edu dataset...")
    data_config = config["data"]
    train_config = config["training"]

    train_dataset = FineWebEduDataset(
        split="train",
        context_length=data_config["context_length"],
        subset=data_config["subset"],
        streaming=True,
        cache_dir=data_config.get("cache_dir"),
        skip_batches=skip_batches,
    )

    # Note: FineWeb Edu sample-10BT only has 'train' split (no validation split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        num_workers=data_config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    print(f"Dataset: {data_config['dataset']}")
    print(f"Context length: {data_config['context_length']}")

    # Optimizer
    opt_config = config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=opt_config["lr"],
        betas=tuple(opt_config["betas"]),
        weight_decay=opt_config["weight_decay"],
        eps=opt_config["eps"],
    )
    print(f"Optimizer: AdamW (lr={opt_config['lr']}, wd={opt_config['weight_decay']})")

    # Scheduler (combining warmup and cosine)
    sched_config = config["scheduler"]

    # Create warmup scheduler
    warmup_scheduler = WarmupLR(
        optimizer,
        warmup_steps=sched_config["warmup_steps"],
    )

    # Create cosine scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config["max_steps"] - sched_config["warmup_steps"],
        eta_min=sched_config["min_lr"],
    )

    # Combine schedulers - use warmup first, then cosine
    class CombinedScheduler:
        def __init__(self, warmup_sched, cosine_sched, warmup_steps):
            self.warmup_sched = warmup_sched
            self.cosine_sched = cosine_sched
            self.warmup_steps = warmup_steps
            self.current_step = 0

        def step(self):
            # Step the appropriate scheduler based on current position
            if self.current_step < self.warmup_steps:
                self.warmup_sched.step()
            else:
                self.cosine_sched.step()
            # Increment AFTER stepping to maintain correct state
            self.current_step += 1

        def state_dict(self):
            return {
                "warmup": self.warmup_sched.state_dict(),
                "cosine": self.cosine_sched.state_dict(),
                "current_step": self.current_step,
            }

        def load_state_dict(self, state_dict):
            self.warmup_sched.load_state_dict(state_dict["warmup"])
            self.cosine_sched.load_state_dict(state_dict["cosine"])
            self.current_step = state_dict["current_step"]

    scheduler = CombinedScheduler(
        warmup_scheduler, cosine_scheduler, sched_config["warmup_steps"]
    )
    print(f"Scheduler: Cosine with {sched_config['warmup_steps']} warmup steps")

    # Callbacks
    # Note: No validation callback since dataset only has train split
    callbacks = [
        CheckpointCallback(
            checkpoint_dir="checkpoints",
            save_every=config["checkpoint"]["save_every"],
            keep_last_n=config["checkpoint"]["keep_last_n"],
            save_best=True,
        ),
        MetricsLoggerCallback(
            log_dir="logs",
            log_every=config["logging"]["log_every"],
        ),
        ThroughputCallback(
            log_every=config["logging"]["log_every"],
            context_length=data_config["context_length"],
            batch_size=train_config["batch_size"]
            * train_config["gradient_accumulation_steps"],
        ),
    ]

    # Create trainer
    grad_clip_norm = None
    if config.get("grad_clip", {}).get("enabled", False):
        grad_clip_norm = config["grad_clip"]["max_norm"]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=str(device),
        context_length=data_config["context_length"],
        grad_accum_steps=train_config["gradient_accumulation_steps"],
        use_amp=train_config.get("use_amp", True),
        callbacks=callbacks,
        scheduler=scheduler,
        grad_clip_norm=grad_clip_norm,
    )

    # Resume from checkpoint if provided
    if resume_path:
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state and sync to resumed step
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Loaded scheduler state from checkpoint")
        else:
            # If no scheduler state saved, manually sync schedulers
            print(f"No scheduler state in checkpoint, syncing manually...")
            resumed_step = checkpoint.get("step", 0)
            scheduler.current_step = resumed_step
            # Fast-forward individual schedulers to correct position
            for _ in range(resumed_step):
                if _ < sched_config["warmup_steps"]:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()
            print(f"Synced schedulers to step {resumed_step}")

        trainer.global_step = checkpoint.get("step", 0)

        # Restore losses if available
        if "losses" in checkpoint:
            trainer.losses = checkpoint["losses"]
            print(f"Restored {len(trainer.losses)} loss values from checkpoint")

        print(f"Resumed from step {trainer.global_step}")

        # Clear checkpoint from memory and free GPU memory
        del checkpoint
        torch.cuda.empty_cache()
        print("Cleared CUDA cache after loading checkpoint")

    return trainer, config


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on FineWeb Edu")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(config.get("seed", 42))

    # Setup training
    print("=" * 80, flush=True)
    print("GPT-2 124M Training on FineWeb Edu 10B Tokens", flush=True)
    print("=" * 80, flush=True)

    trainer, config = setup_training(config, resume_path=args.resume)

    # Train
    print("\nStarting training...", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()
    trainer.train(
        epochs=100,  # Large number, will stop at max_steps
        log_interval=config["logging"]["log_every"],
        max_steps=config["training"]["max_steps"],
    )

    # Save final results
    training_time = time.time() - start_time

    # Note: No validation loss tracked since dataset only has train split
    try:
        final_train_loss = trainer.losses[-1] if trainer.losses else None
    except (StopIteration, IndexError):
        final_train_loss = None

    results = {
        "model": "gpt2-124m",
        "dataset": "fineweb-edu-10b",
        "final_train_loss": final_train_loss,
        "final_perplexity": (
            torch.exp(torch.tensor(final_train_loss)).item()
            if final_train_loss
            else None
        ),
        "total_steps": trainer.global_step,
        "total_tokens": trainer.global_step
        * config["training"]["batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * config["data"]["context_length"],
        "training_time_hours": training_time / 3600,
        "config": config,
    }

    results_path = Path("results") / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete!")
    print(
        f"Final training loss: {final_train_loss:.4f}"
        if final_train_loss
        else "Training complete!"
    )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
