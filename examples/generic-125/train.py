#!/usr/bin/env python3
"""
Generic 125M Training on FineWeb Edu 10B Tokens

Target: Achieve 3.28 validation loss

Usage:
    python -u train.py --epochs 1 --batch_size 16 --learning_rate 3e-4
    python -u train.py --resume checkpoints/checkpoint_latest.pt
    python -u train.py --epochs 5 --save_every 1000 --eval_every 500

    Note: Use -u flag for unbuffered output when redirecting to log files:
    nohup python -u train.py --epochs 3 > logs/train.log 2>&1 &
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from olm.data.datasets.fineweb_edu import FineWebEduDataset
from olm.data.datasets import DataLoader
from olm.train.trainer import Trainer
from olm.nn.blocks import LM
from olm.train.optim import AdamW


class TrainingLogger:
    """Handles logging to console and file"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{timestamp}.log"
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"

    def log(self, message, level="INFO"):
        """Log to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

    def log_metrics(self, step, metrics):
        """Log metrics to JSONL file"""
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")


class CheckpointManager:
    """Manages model checkpoints with early stopping"""

    def __init__(self, checkpoint_dir, keep_best=3, patience=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.keep_best = keep_best
        self.patience = patience
        self.best_loss = float("inf")
        self.best_checkpoints = []
        self.steps_without_improvement = 0

    def save_checkpoint(self, model, optimizer, step, loss, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Check if this is a best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
            self._save_best_checkpoint(checkpoint, step, loss)
            return False  # Not early stopping
        else:
            self.steps_without_improvement += 1
            return self.steps_without_improvement >= self.patience

    def _save_best_checkpoint(self, checkpoint, step, loss):
        """Save and manage best checkpoints"""
        best_path = (
            self.checkpoint_dir / f"checkpoint_best_step_{step}_loss_{loss:.4f}.pt"
        )
        torch.save(checkpoint, best_path)

        self.best_checkpoints.append((loss, best_path))
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.keep_best:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint


class MetricsTracker:
    """Tracks and visualizes training metrics"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.steps = []
        self.losses = []
        self.learning_rates = []
        self.throughputs = []

    def add_metrics(self, step, loss, lr=None, throughput=None):
        """Add metrics for a step"""
        self.steps.append(step)
        self.losses.append(loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if throughput is not None:
            self.throughputs.append(throughput)

    def save_plots(self):
        """Save training plots"""
        if not self.steps:
            return

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Learning rate plot
        if self.learning_rates:
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.learning_rates, label="Learning Rate")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                self.output_dir / "learning_rate.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        # Throughput plot
        if self.throughputs:
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.throughputs, label="Throughput (tokens/sec)")
            plt.xlabel("Step")
            plt.ylabel("Tokens/sec")
            plt.title("Training Throughput")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                self.output_dir / "throughput.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

    def save_summary(self):
        """Save training summary"""
        if not self.steps:
            return

        summary = {
            "total_steps": len(self.steps),
            "final_loss": self.losses[-1] if self.losses else None,
            "best_loss": min(self.losses) if self.losses else None,
            "avg_loss": sum(self.losses) / len(self.losses) if self.losses else None,
            "avg_throughput": (
                sum(self.throughputs) / len(self.throughputs)
                if self.throughputs
                else None
            ),
        }

        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Generic 125M model on FineWeb Edu"
    )

    # Model config (fixed for this experiment)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_multiplier", type=float, default=4.0)

    # Training config
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )

    # Checkpointing and logging
    parser.add_argument(
        "--save_every", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_every", type=int, default=100, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log metrics every N steps"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    # Early stopping
    parser.add_argument(
        "--early_stopping_patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--keep_best", type=int, default=3, help="Keep N best checkpoints"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize logging and tracking
    logger = TrainingLogger(args.log_dir)
    checkpoint_manager = CheckpointManager(
        args.checkpoint_dir,
        keep_best=args.keep_best,
        patience=args.early_stopping_patience,
    )
    metrics_tracker = MetricsTracker(args.results_dir)

    logger.log("=" * 80)
    logger.log("Generic 125M Training on FineWeb Edu 10B Tokens")
    logger.log("=" * 80)

    # Log configuration
    logger.log(f"Configuration:")
    for key, value in vars(args).items():
        logger.log(f"  {key}: {value}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"\nDevice: {device}")
    if device == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Initialize model
    logger.log("\nInitializing model...")
    model = LM(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        ff_multiplier=args.ff_multiplier,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), args.learning_rate)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        logger.log(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        logger.log(f"Resumed from step {start_step}")

    # Initialize dataset and dataloader
    logger.log("\nInitializing dataset...")
    dataset = FineWebEduDataset(
        split="train",
        context_length=args.context_length,
        subset="sample-10BT",
        streaming=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    logger.log(f"Dataset: FineWeb Edu sample-10BT")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Context length: {args.context_length}")

    # Initialize trainer
    logger.log("\nInitializing trainer...")
    trainer = Trainer(
        model, optimizer, dataloader, device, args.context_length, use_amp=args.use_amp
    )

    # Training loop
    logger.log("\n" + "=" * 80)
    logger.log("Starting training...")
    logger.log("=" * 80 + "\n")

    try:
        step = start_step
        start_time = time.time()

        for epoch in range(args.epochs):
            logger.log(f"Epoch {epoch + 1}/{args.epochs}")

            # Train for one epoch
            # Note: Using a large number for steps_per_epoch since we're streaming
            losses = trainer.train(
                epochs=1,
                steps_per_epoch=100000,  # Large number for streaming
                log_interval=args.log_every,
            )

            # Track metrics for each step
            for i, loss in enumerate(losses):
                step += 1

                # Calculate throughput
                elapsed = time.time() - start_time
                tokens_processed = step * args.batch_size * args.context_length
                throughput = tokens_processed / elapsed if elapsed > 0 else 0

                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                # Track metrics
                metrics_tracker.add_metrics(step, loss, current_lr, throughput)

                # Log metrics
                if step % args.log_every == 0:
                    logger.log(
                        f"Step {step}: Loss={loss:.4f}, LR={current_lr:.6f}, "
                        f"Throughput={throughput:.0f} tok/s"
                    )
                    logger.log_metrics(
                        step,
                        {
                            "loss": loss,
                            "learning_rate": current_lr,
                            "throughput": throughput,
                        },
                    )

                # Save checkpoint
                if step % args.save_every == 0:
                    logger.log(f"Saving checkpoint at step {step}...")
                    early_stop = checkpoint_manager.save_checkpoint(
                        model,
                        optimizer,
                        step,
                        loss,
                        metrics={"learning_rate": current_lr, "throughput": throughput},
                    )

                    if early_stop:
                        logger.log(
                            f"Early stopping triggered after {checkpoint_manager.steps_without_improvement} "
                            f"steps without improvement"
                        )
                        logger.log(f"Best loss: {checkpoint_manager.best_loss:.4f}")
                        raise KeyboardInterrupt  # Use this to break out cleanly

                # Save plots periodically
                if step % (args.save_every * 2) == 0:
                    logger.log("Saving training plots...")
                    metrics_tracker.save_plots()
                    metrics_tracker.save_summary()

    except KeyboardInterrupt:
        logger.log("\nTraining interrupted by user or early stopping")

    except Exception as e:
        logger.log(f"\nError during training: {str(e)}", level="ERROR")
        raise

    finally:
        # Final checkpoint and metrics
        logger.log("\n" + "=" * 80)
        logger.log("Training complete!")
        logger.log("=" * 80)

        if step > start_step:
            logger.log(f"\nSaving final checkpoint...")
            checkpoint_manager.save_checkpoint(
                model, optimizer, step, losses[-1] if losses else float("inf")
            )

            logger.log("Saving final plots and summary...")
            metrics_tracker.save_plots()
            metrics_tracker.save_summary()

            # Print summary
            logger.log(f"\nTraining Summary:")
            logger.log(f"  Total steps: {step}")
            logger.log(
                f"  Final loss: {losses[-1]:.4f}" if losses else "  No losses recorded"
            )
            logger.log(f"  Best loss: {checkpoint_manager.best_loss:.4f}")
            logger.log(f"  Checkpoints saved to: {args.checkpoint_dir}")
            logger.log(f"  Logs saved to: {args.log_dir}")
            logger.log(f"  Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
