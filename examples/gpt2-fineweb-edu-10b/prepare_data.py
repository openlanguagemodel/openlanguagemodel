#!/usr/bin/env python3
"""
Prepare and validate FineWeb Edu dataset for training.

This script:
1. Downloads and caches the dataset
2. Validates tokenizer compatibility
3. Estimates dataset statistics
4. Tests data loading pipeline

Usage:
    python prepare_data.py --cache_dir ./data_cache
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from olm.data.datasets.fineweb_edu import FineWebEduDataset
from transformers import GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb Edu dataset")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data_cache",
        help="Directory to cache dataset",
    )
    parser.add_argument(
        "--context_length", type=int, default=1024, help="Context length for sequences"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to test"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FineWeb Edu Dataset Preparation")
    print("=" * 80)

    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"   Tokenizer type: {type(tokenizer).__name__}")

    # Create dataset
    print("\n2. Loading training dataset...")
    train_dataset = FineWebEduDataset(
        split="train",
        context_length=args.context_length,
        subset="sample-10BT",
        streaming=True,
        cache_dir=args.cache_dir,
    )
    print(f"   Context length: {args.context_length}")
    print(f"   Subset: sample-10BT (10B tokens)")

    # Test data loading
    print(f"\n3. Testing data loading ({args.num_samples} samples)...")
    train_iter = iter(train_dataset)

    for i in range(args.num_samples):
        input_ids, labels = next(train_iter)

        if i == 0:
            print(f"\n   Sample shape:")
            print(f"   - Input IDs: {input_ids.shape} (dtype: {input_ids.dtype})")
            print(f"   - Labels: {labels.shape} (dtype: {labels.dtype})")

            # Decode first few tokens
            print(f"\n   First sequence (first 50 tokens):")
            decoded = tokenizer.decode(input_ids[:50])
            print(f"   {decoded[:200]}...")

            # Verify shift (labels are input shifted by 1)
            # In a sequence [a, b, c, d, e], if input_ids = [a, b, c, d], labels = [b, c, d, e]
            # So input_ids[i+1] == labels[i] for all valid i
            print(f"\n   Verifying label shift:")
            print(
                f"   - input_ids[1] == labels[0]: {input_ids[1].item() == labels[0].item()}"
            )
            print(
                f"   - input_ids[-1] == labels[-2]: {input_ids[-1].item() == labels[-2].item()}"
            )

        # Check for valid token IDs
        assert input_ids.min() >= 0 and input_ids.max() < tokenizer.vocab_size
        assert labels.min() >= 0 and labels.max() < tokenizer.vocab_size

        if (i + 1) % 5 == 0:
            print(f"   Loaded {i + 1}/{args.num_samples} samples...")

    print(f"\n✓ Successfully loaded {args.num_samples} samples")

    # Note: FineWeb Edu sample-10BT only has 'train' split
    print("\n4. Note: This dataset only has a 'train' split (no validation split)")

    # Estimate dataset size
    print("\n5. Dataset statistics:")
    total_tokens = 10_000_000_000  # 10B tokens
    sequences = total_tokens // args.context_length
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Sequences (context={args.context_length}): {sequences:,}")
    print(f"   Estimated batches (batch_size=16): {sequences // 16:,}")

    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)
    print(f"\nCache directory: {args.cache_dir}")
    print("Ready to start training with train.py")

    # Clean up dataset iterators to prevent cleanup errors
    del train_iter
    del train_dataset

    # Force clean exit to avoid threading cleanup issues with HuggingFace datasets
    os._exit(0)


if __name__ == "__main__":
    main()
