"""
Example: Training with Weights & Biases (wandb) integration.

This script demonstrates how to use WandBCallback for experiment tracking,
including metrics logging, alerts, and hyperparameter sweeps.
"""

import torch
from olm.models.meta.llama2 import Llama2Model
from olm.data.datasets import DataLoader, FineWebEduDataset
from olm.data.tokenization import HFTokenizer
from olm.train.trainer import Trainer
from olm.logging import WandBCallback, create_sweep, get_sweep_config_template


def train_with_wandb():
    """Basic training with wandb logging."""

    # Initialize tokenizer and dataset
    tokenizer = HFTokenizer("gpt2")
    dataset = FineWebEduDataset(
        tokenizer=tokenizer,
        subset="sample-10BT",
        context_length=512,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
    )

    # Create model
    model = Llama2Model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=512,
        intermediate_size=2048,
        num_layers=8,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=512,
    )

    # Create WandB callback with all features
    wandb_callback = WandBCallback(
        project="olm-training",
        name="llama2-8layer-baseline",
        tags=["llama2", "baseline", "8-layer"],
        notes="Baseline training run with 8-layer Llama2 model",
        # Logging configuration
        log_frequency=10,  # Log every 10 steps
        log_gradients=True,  # Log gradient histograms
        log_model=True,  # Save checkpoints as artifacts
        watch_model=True,  # Use wandb.watch for automatic tracking
        log_system_metrics=True,  # Log GPU/CPU metrics
        # Alert configuration
        alert_thresholds={
            "train/loss": {"max": 10.0},  # Alert if loss > 10
            "train/learning_rate": {"min": 1e-7},  # Alert if LR too low
        },
        # Optional: Offline mode for air-gapped environments
        # offline=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW,
        dataloader=dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        context_length=512,
        learning_rate=3e-4,
        weight_decay=0.1,
        grad_accum_steps=4,
        grad_clip_norm=1.0,
        callbacks=[wandb_callback],
    )

    # Train
    trainer.train(epochs=3, max_steps=1000, log_interval=10)

    print("\n✅ Training complete! View results at: https://wandb.ai")


def train_with_predictions_logging():
    """Example with prediction table logging."""

    # ... (setup same as above)

    wandb_callback = WandBCallback(
        project="olm-training",
        name="llama2-with-predictions",
        log_predictions=True,  # Enable prediction logging
    )

    # In your training loop, you can log predictions:
    # wandb_callback.log_predictions(
    #     step=trainer.global_step,
    #     inputs=["Hello world"],
    #     predictions=["predicted text"],
    #     targets=["target text"],
    # )


def hyperparameter_sweep():
    """Example: Hyperparameter sweep with wandb."""

    # Get sweep config template
    sweep_config = get_sweep_config_template(method="bayes")

    # Customize sweep parameters
    sweep_config["parameters"] = {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "batch_size": {
            "values": [8, 16, 32],
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.3,
        },
        "num_layers": {
            "values": [4, 8, 12],
        },
    }

    # Create sweep
    sweep_id = create_sweep(
        sweep_config=sweep_config,
        project="olm-training",
    )

    print(f"Sweep created! Run: wandb agent {sweep_id}")

    # To run the sweep, define a train function that uses wandb.config:
    def sweep_train():
        import wandb

        # Initialize wandb (will be done by agent)
        run = wandb.init()

        # Get hyperparameters from sweep
        config = wandb.config

        # Setup model with sweep parameters
        tokenizer = HFTokenizer("gpt2")
        dataset = FineWebEduDataset(tokenizer=tokenizer, context_length=512)
        dataloader = DataLoader(dataset, batch_size=config.batch_size)

        model = Llama2Model(
            vocab_size=tokenizer.vocab_size,
            embed_dim=512,
            intermediate_size=2048,
            num_layers=config.num_layers,
            num_heads=8,
            num_kv_heads=8,
            max_seq_len=512,
        )

        # WandB callback will use the existing run
        wandb_callback = WandBCallback(
            project="olm-training",
            reinit=False,  # Don't create new run, use existing
        )

        trainer = Trainer(
            model=model,
            optimizer=torch.optim.AdamW,
            dataloader=dataloader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            context_length=512,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            callbacks=[wandb_callback],
        )

        trainer.train(epochs=1, max_steps=500)

    # Run the sweep with: wandb agent <sweep_id>
    # Or programmatically:
    # wandb.agent(sweep_id, function=sweep_train, count=10)


def distributed_training_with_wandb():
    """Example: Distributed training with wandb (DDP/FSDP)."""
    from olm.core.dist import setup_distributed, get_local_rank
    from olm.train.trainer import DDPTrainer

    # Setup distributed
    setup_distributed()

    # Initialize components
    tokenizer = HFTokenizer("gpt2")
    dataset = FineWebEduDataset(tokenizer=tokenizer, context_length=512)
    dataloader = DataLoader(dataset, batch_size=16, distributed=True)

    model = Llama2Model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=512,
        intermediate_size=2048,
        num_layers=8,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=512,
    )

    # WandB callback (automatically handles distributed - only rank 0 logs)
    wandb_callback = WandBCallback(
        project="olm-distributed-training",
        name="ddp-4gpu-run",
        tags=["ddp", "distributed"],
        log_frequency=10,
        log_model=True,
    )

    # Create DDP trainer
    trainer = DDPTrainer(
        model=model,
        optimizer=torch.optim.AdamW,
        dataloader=dataloader,
        device=f"cuda:{get_local_rank()}",
        context_length=512,
        learning_rate=3e-4,
        callbacks=[wandb_callback],
    )

    trainer.train(epochs=3, max_steps=1000)

    print("\n✅ Distributed training complete!")


if __name__ == "__main__":
    # Choose which example to run:

    # Basic training with wandb
    train_with_wandb()

    # Training with prediction logging
    # train_with_predictions_logging()

    # Hyperparameter sweep
    # hyperparameter_sweep()

    # Distributed training
    # Run with: torchrun --nproc_per_node=4 wandb_example.py
    # distributed_training_with_wandb()
