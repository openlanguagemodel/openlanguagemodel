"""
Example: Automatic Trainer Selection with AutoTrainer

This example demonstrates how to use AutoTrainer for automatic device
detection and optimal trainer selection (Trainer, DDPTrainer, or FSDPTrainer).

Run:
    # Single GPU:
    python auto_trainer_example.py

    # Multi-GPU with DDP/FSDP (automatic selection):
    torchrun --nproc_per_node=4 auto_trainer_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path for importing olm
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from olm.models.openai import GPT2
from olm.data.tokenization import HFTokenizer
from olm.data.datasets import LocalTextDataset, DataLoader
from olm.train import AutoTrainer, detect_devices, estimate_model_size
from olm.train.optim import AdamW


def main():
    print("=" * 80)
    print("AutoTrainer Example: Automatic Device Detection & Trainer Selection")
    print("=" * 80)

    # ========================================================================
    # Example 1: Basic Auto Mode (Simplest Usage)
    # ========================================================================
    print("\n[Example 1] Basic Auto Mode")
    print("-" * 80)

    # Create a simple model
    model = GPT2()
    print(f"Model created: GPT2 (124M parameters)")

    # Create tokenizer and dataset
    tokenizer = HFTokenizer("gpt2")
    dataset = LocalTextDataset(
        location="./data/sample_texts",  # Replace with your data path
        tokenizer=tokenizer,
        context_length=512,
        shuffle=True,
    )
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    # AutoTrainer with device="auto" - that's all you need!
    trainer = AutoTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device="auto",  # Magic! Automatically detects and configures
        context_length=512,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    print("\nTrainer initialized successfully!")
    print("The system automatically selected the optimal trainer type based on:")
    print("  - Number of available GPUs")
    print("  - Model size")
    print("  - Available GPU memory")

    # Train
    # trainer.train(epochs=1, log_interval=10)

    # ========================================================================
    # Example 2: Memory-Efficient Preset
    # ========================================================================
    print("\n[Example 2] Memory-Efficient Preset")
    print("-" * 80)

    # For large models, use memory_efficient preset
    # This prioritizes FSDP with CPU offload
    trainer_mem = AutoTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device="auto",
        preset="memory_efficient",  # Prioritize memory over speed
        context_length=512,
        learning_rate=3e-4,
    )

    print("\nMemory-efficient trainer configured!")
    print("This configuration prioritizes:")
    print("  - FSDP over DDP (when multi-GPU)")
    print("  - CPU offload for parameters")
    print("  - Optimal sharding strategies")

    # ========================================================================
    # Example 3: Speed-Optimized Preset
    # ========================================================================
    print("\n[Example 3] Speed-Optimized Preset")
    print("-" * 80)

    trainer_speed = AutoTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device="auto",
        preset="speed",  # Prioritize speed over memory
        context_length=512,
        learning_rate=3e-4,
    )

    print("\nSpeed-optimized trainer configured!")
    print("This configuration prioritizes:")
    print("  - DDP over FSDP (when model fits)")
    print("  - No CPU offload")
    print("  - Larger communication buckets")

    # ========================================================================
    # Example 4: Manual Device Detection
    # ========================================================================
    print("\n[Example 4] Manual Device Detection")
    print("-" * 80)

    # You can also detect devices manually for inspection
    from olm.train import DeviceConfig, TrainerStrategy

    config = detect_devices(verbose=True)

    # Optionally modify the config
    if config.num_gpus > 1:
        print("\nMulti-GPU detected. You could force a specific strategy:")
        print(f"  - TrainerStrategy.MULTI_GPU_DDP")
        print(f"  - TrainerStrategy.MULTI_GPU_FSDP_HYBRID")
        print(f"  - TrainerStrategy.MULTI_GPU_FSDP_FULL")

    # ========================================================================
    # Example 5: Force Specific Strategy
    # ========================================================================
    print("\n[Example 5] Force Specific Strategy")
    print("-" * 80)

    # Force DDP even if FSDP might be selected
    trainer_forced = AutoTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device="auto",
        force_strategy=TrainerStrategy.MULTI_GPU_DDP if config.num_gpus > 1 else None,
        context_length=512,
        learning_rate=3e-4,
    )

    print("\nTrainer with forced strategy configured!")

    # ========================================================================
    # Example 6: Estimate Model Memory
    # ========================================================================
    print("\n[Example 6] Model Memory Estimation")
    print("-" * 80)

    memory_info = estimate_model_size(model, verbose=True)

    print("\nYou can use this information to:")
    print("  - Decide if your model fits on available GPUs")
    print("  - Choose between DDP and FSDP")
    print("  - Determine gradient accumulation steps")

    # ========================================================================
    # Example 7: Backward Compatible Usage
    # ========================================================================
    print("\n[Example 7] Backward Compatible Usage")
    print("-" * 80)

    # Old code still works! AutoTrainer is backward compatible
    # These will use single-device trainer
    trainer_cuda = AutoTrainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        context_length=512,
        learning_rate=3e-4,
    )

    print("\nBackward compatible mode:")
    print("  - device='cuda' -> Single GPU Trainer")
    print("  - device='cuda:0' -> Single GPU Trainer on device 0")
    print("  - device='cpu' -> CPU Trainer")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary: AutoTrainer Usage Patterns")
    print("=" * 80)
    print("\n1. Simplest (Recommended):")
    print("   trainer = AutoTrainer(model=model, device='auto', ...)")
    print("\n2. With Preset:")
    print(
        "   trainer = AutoTrainer(model=model, device='auto', preset='memory_efficient', ...)"
    )
    print("\n3. Force Strategy:")
    print(
        "   trainer = AutoTrainer(model=model, device='auto', force_strategy=TrainerStrategy.MULTI_GPU_DDP, ...)"
    )
    print("\n4. Backward Compatible:")
    print("   trainer = AutoTrainer(model=model, device='cuda', ...)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
