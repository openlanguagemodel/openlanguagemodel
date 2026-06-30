# GPT-2 124M Training on FineWeb Edu 10B Tokens

Training GPT-2 124M on the FineWeb Edu dataset (10B token sample).

## Quick Start

```bash
# 1. Prepare data (downloads and validates dataset)
python prepare_data.py --cache_dir ./data_cache

# 2. Start training
python train.py --config config.yaml

# 3. Resume from checkpoint (if interrupted)
python train.py --config config.yaml --resume checkpoints/step_5000.pt
```

## Multi-GPU Training

```bash
# Using torchrun for distributed training (4 GPUs)
torchrun --nproc_per_node=4 train.py --config config.yaml
```

## Configuration

Edit `config.yaml` to adjust:

-   Model architecture (embed_dim, num_layers, etc.)
-   Training hyperparameters (learning rate, batch size, etc.)
-   Hardware settings (num_gpus, use_amp, etc.)

## Monitoring

Training logs are saved to:

-   `logs/training.log` - Text logs
-   `logs/metrics_*.jsonl` - Structured metrics (JSON lines)

## Results

Final results will be saved to:

-   `results/final_results.json` - Complete training metrics
-   `checkpoints/best_model.pt` - Best validation loss checkpoint
-   `checkpoints/step_*.pt` - Regular checkpoints

## Expected Performance

With the default configuration:

-   **Training objective**: causal next-token prediction on FineWeb-Edu
-   **Training time**: ~8-12 hours on 4x A100 GPUs
-   **Throughput**: ~300K-400K tokens/second
-   **Total tokens**: 10 billion
-   **Total steps**: ~19,073 steps

## Hardware Requirements

**Minimum:**

-   1x GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090)
-   32GB system RAM
-   100GB disk space (for dataset cache)

**Recommended:**

-   4x A100 40GB GPUs
-   128GB system RAM
-   NVMe SSD for dataset cache

## Dataset

**FineWeb Edu** (HuggingFaceFW/fineweb-edu):

-   High-quality educational web content
-   Sample: 10 billion tokens
-   Tokenizer: GPT-2 BPE
-   Vocab size: 50,257

## Model Architecture

**GPT-2 124M:**

-   Embedding dimension: 768
-   Layers: 12
-   Attention heads: 12
-   Context length: 1024
-   Parameters: ~124 million

## Training Details

**Optimization:**

-   Optimizer: AdamW
-   Learning rate: 6e-4 with cosine decay
-   Warmup: 1,000 steps (~5% of training)
-   Weight decay: 0.1
-   Gradient clipping: 1.0

**Batch Configuration:**

-   Per-device batch size: 16
-   Gradient accumulation: 8 steps
-   Effective batch (4 GPUs): 512 sequences = 524,288 tokens

**Mixed Precision:**

-   Type: bfloat16
-   Automatic mixed precision (AMP) enabled

## Checkpointing

Checkpoints are saved:

-   Every 1,000 steps
-   On best validation loss
-   Includes model, optimizer, scheduler, and scaler states

## Troubleshooting

**Out of memory:**

-   Reduce `batch_size` in config.yaml
-   Increase `gradient_accumulation_steps`
-   Disable `use_amp` (slower but less memory)

**Slow training:**

-   Ensure `pin_memory: true`
-   Increase `num_workers` for data loading
-   Enable `compile: true` (requires PyTorch 2.0+)

**Dataset download issues:**

-   Check internet connection
-   Ensure sufficient disk space
-   Try clearing cache_dir and redownloading

## Citation

```bibtex
@misc{fineweb-edu,
  title={FineWeb-Edu},
  author={HuggingFace Team},
  year={2024},
  howpublished={\\url{https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu}}
}

@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```
