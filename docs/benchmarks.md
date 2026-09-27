---
description: Measured OLM training throughput, MFU, peak memory and cost per billion tokens on single rented GPUs, multi-GPU scaling on A100s, and how to reproduce or contribute a GPU.
---

# GPU Benchmarks

How fast does an OLM model train on one GPU, and what does a billion training tokens cost? This page answers with measurements, not estimates: every measured number in the tables below is generated from the [benchmark catalog](../benchmarks/throughput/data/benchmark_catalog.csv) by [`render_docs_tables.py`](../benchmarks/throughput/render_docs_tables.py), and the test suite fails if the page and the catalog disagree.

For every GPU, model size (150M to 8B) and context length (512 to 32,768 tokens), with any token budget and price, use the [interactive explorer on the website](https://openlanguagemodel.github.io/openlanguagemodel/benchmarks/).

> [!NOTE]
> These are steady-state measurements of a full training step on synthetic tokens on a single rented GPU. They leave out data loading, evaluation, checkpointing, restarts and idle time, so a real run takes longer and costs more. Use them to compare GPUs and to set a lower bound on a budget.

## What is measured

Each timed step is a complete training step for a dense, tied-embedding Llama-3-style model built with `olm.models.meta.llama3.Llama3Model`:

1. a forward pass under BF16 autocast, with FP32 parameters, gradients and AdamW state;
2. shifted-token cross-entropy;
3. the backward pass;
4. a fused AdamW update.

There is no activation checkpointing and no `torch.compile`; attention uses PyTorch SDPA. For every model the harness sweeps the batch size in powers of two at 2,048 tokens, then extends the sweep to shorter and longer contexts. Each configuration runs an adaptive warmup until step time is stable, then 20 measured steps, and the whole protocol runs twice (seeds 11 and 22). The full protocol is in the [methodology README](../benchmarks/throughput/README.md).

## Single GPU: 150M and 500M at 2,048 tokens

<!-- BEGIN GENERATED: snapshot -->
Catalog: 1,342 rows covering 5 GPUs (RTX 4090, RTX 5090, A100 PCIe 80GB, RTX PRO 6000, H100 SXM 80GB), 9 model sizes from 150M to 8B, contexts from 512 to 32,768 tokens and seeds 11 and 22. Prices: RunPod on-demand rates captured on 2026-07-24.
<!-- END GENERATED: snapshot -->

### 150M parameters

<!-- BEGIN GENERATED: single-gpu-150m -->
| GPU | Best batch | Tokens/s | MFU (spec / board clock) | Peak VRAM | $ per 1B tokens, Community | $ per 1B tokens, Secure |
|---|--:|--:|--:|--:|--:|--:|
| RTX 4090 (24 GB) | 4 | 58,130 | 45.0% / 36.2% | 12.1 GB | $1.62 | $3.30 |
| RTX 5090 (32 GB) | 8 | 87,013 | 53.2% / 41.4% | 22.1 GB | $2.20 | $3.16 |
| A100 PCIe 80GB | 16 | 68,758 | 28.2% / 28.2% | 42.0 GB | $4.81 | $5.62 |
| RTX PRO 6000 (96 GB) | 8 | 101,657 | 27.8% / 27.8% | 22.1 GB | $4.62 | $5.44 |
| H100 SXM 80GB | 16 | 139,706 | 18.1% / 18.1% | 42.1 GB | $5.35 | $5.95 |
<!-- END GENERATED: single-gpu-150m -->

### 500M parameters

<!-- BEGIN GENERATED: single-gpu-500m -->
| GPU | Best batch | Tokens/s | MFU (spec / board clock) | Peak VRAM | $ per 1B tokens, Community | $ per 1B tokens, Secure |
|---|--:|--:|--:|--:|--:|--:|
| RTX 4090 (24 GB) | 2 | 21,361 | 50.2% / 40.4% | 16.4 GB | $4.42 | $8.97 |
| RTX 5090 (32 GB) | 4 | 33,158 | 61.4% / 47.9% | 25.6 GB | $5.78 | $8.29 |
| A100 PCIe 80GB | 16 | 28,762 | 35.8% / 35.8% | 81.1 GB | $11.49 | $13.42 |
| RTX PRO 6000 (96 GB) | 4 | 41,283 | 34.2% / 34.2% | 25.6 GB | $11.37 | $13.39 |
| H100 SXM 80GB | 16 | 63,145 | 24.8% / 24.8% | 81.2 GB | $11.83 | $13.15 |
<!-- END GENERATED: single-gpu-500m -->

<!-- BEGIN GENERATED: takeaways -->
- **150M:** the H100 SXM 80GB is fastest (139,706 tokens/s). The cheapest per token is the RTX 4090 on Community ($1.62 per 1B tokens) and the RTX 5090 on Secure ($3.16 per 1B tokens).
- **500M:** the H100 SXM 80GB is fastest (63,145 tokens/s). The cheapest per token is the RTX 4090 on Community ($4.42 per 1B tokens) and the RTX 5090 on Secure ($8.29 per 1B tokens).
<!-- END GENERATED: takeaways -->

How to read the columns:

- **Best batch** is the batch size with the highest median tokens/s among the batches that fit for both seeds. A batch that fit one seed and ran out of memory on the other is never chosen.
- **Tokens/s** is computed from each seed's median step time over the measured steps; the table shows the median of the two seeds.
- **MFU** (model FLOP utilization) is achieved training FLOPs divided by the GPU's dense BF16 peak (its dense FP16 peak for a GPU run with `--amp-dtype fp16`). Training FLOPs per token are `6N + 12 · layers · width · context`, where `N` counts unique parameters. *Spec* uses NVIDIA's published dense peak at the reference boost clock. *Board clock* rescales that peak to the maximum SM clock the rented board reported, which is higher than the reference clock on the RTX 4090 and RTX 5090 in this catalog. Sparse "up to" figures are never used.
- **Peak VRAM** is `torch.cuda.max_memory_allocated()` for the whole step, in GB (10⁹ bytes). The allocator reserves more than this.
- **$ per 1B tokens** is `10⁹ / tokens_per_second / 3,600 × USD per GPU-hour`, using the dated RunPod on-demand rate for each cloud tier. It is compute only: storage, setup, idle time and taxes are excluded, and prices change.

MFU and speed rank GPUs differently. A small model keeps a consumer card busier than it keeps an H100, but the H100 still finishes first. Pick a GPU by tokens/s and cost; read MFU as a measure of how much of the hardware the step uses.

## Multi-GPU scaling

The OLM paper ([arXiv:2607.16669](https://arxiv.org/abs/2607.16669)) measures weak scaling with `torch.nn.parallel.DistributedDataParallel` on one NVLink node of A100-SXM4-80GB GPUs. The model is the 348M-parameter Llama-3-style configuration (1,024 wide, 28 layers, 16 query and 4 key/value heads, 32,000-token vocabulary, tied embeddings), the same shape as the 350M model in the single-GPU catalog. Each GPU processes one 2,048-token sequence per step under BF16 autocast with non-fused AdamW, SDPA attention and no `torch.compile`; each run has 50 warmup and 200 measured steps, and every GPU count ran three times.

<!-- BEGIN GENERATED: scaling -->
| GPUs | Tokens/s (mean ± std) | Tokens/s per GPU | Scaling efficiency | Peak memory per GPU |
|---|--:|--:|--:|--:|
| 1 × A100-SXM4-80GB | 17,785 ± 104 | 17,785 | 100.0% | 9.06 GB |
| 2 × A100-SXM4-80GB | 32,426 ± 451 | 16,213 | 91.2% | 10.47 GB |
| 4 × A100-SXM4-80GB | 64,427 ± 419 | 16,107 | 90.6% | 10.47 GB |
<!-- END GENERATED: scaling -->

Scaling efficiency is `tokens_per_second(N) / (N × tokens_per_second(1))`, from the mean of the three runs. Almost all of the loss comes from going from one GPU to two, where DDP starts all-reducing gradients every step; going from two to four GPUs costs less than one more point. Peak memory is `torch.cuda.max_memory_allocated()` on the reporting rank.

These runs used a different A100 variant (SXM, not PCIe), batch 1 per GPU and a non-fused optimizer, so do not compare them directly with the single-GPU table. The summary, with every replicate, is [`multi_gpu_scaling.json`](../benchmarks/throughput/data/multi_gpu_scaling.json).

## Look up any point in the catalog

No GPU is needed to query the catalog. From the repository root:

```bash
python benchmarks/throughput/lookup_training_point.py \
    --gpu "RTX 5090" --model 500m --context 2048
```

It prints the best batch, the per-seed tokens/s, board-clock MFU and peak VRAM, and the first batch that ran out of memory.

## Reproduce a measurement

You need a CUDA GPU with `nvidia-smi` and OLM installed from a clone of this repository (`pip install -e .`). The publication suite runs both seeds of the batch sweep, the context sweep, the validators, the exports and a 200 ms GPU telemetry trace, and stops rather than overwrite a partial result:

```bash
out=$HOME/olm-bench/my-gpu
benchmarks/throughput/capture_environment_manifest.sh \
    "$out/manifests/environment" "$(command -v python)" "$PWD"
benchmarks/throughput/run_publication_suite.sh \
    my_gpu "$out" "$(command -v python)" "$PWD"
```

MFU needs the GPU's dense peak and the clock it is specified at. Both are registered for the GPUs in the catalog and for the RTX A6000, RTX 6000 Ada, H200, B200, B300 and T4. For any other GPU, add its dense BF16 peak, reference boost clock and NVIDIA source to `KNOWN_DENSE_BF16_TFLOPS` in `benchmark_training_mfu.py` and `GPU_REFERENCES` in `annotate_clock_adjusted_mfu.py` before running. Setting `MODELS=150m,250m` limits the 2,048-token sweep and the main context sweep to a subset of models.

When the suite finishes, `$out/results/` holds the raw JSON for every stage and two CSV exports in the catalog's column layout. The [methodology README](../benchmarks/throughput/README.md#run-it) explains each file.

## Contribute a GPU

The catalog grows one GPU at a time, and the Tesla T4 is next: it is the GPU in a free Google Colab session, so anyone can add it. The T4 has no BF16 Tensor Cores, so it runs the harness with `--amp-dtype fp16`: the same step under FP16 autocast with dynamic loss scaling, with MFU measured against the T4's 65 TFLOP/s dense FP16 peak. The website lists the T4 as *not yet measured* until those rows land.

In a Colab notebook with a T4 runtime, mount Google Drive in the first cell so the results survive a disconnect:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then run the suite in a second cell:

```bash
%%bash
set -e
[ -d openlanguagemodel ] || git clone https://github.com/openlanguagemodel/openlanguagemodel
cd openlanguagemodel
pip install -q -e .
out=/content/drive/MyDrive/olm-bench/t4
benchmarks/throughput/capture_environment_manifest.sh \
    "$out/manifests/environment" "$(command -v python)" "$PWD"
AMP_DTYPE=fp16 benchmarks/throughput/run_publication_suite.sh \
    t4 "$out" "$(command -v python)" "$PWD"
```

The runner skips stages that already completed, so after a disconnect you can run the same cell again. It refuses to overwrite a stage that stopped halfway: move that stage's JSON out of `results/` first.

Then, in a clone of the repository (the Colab one, or a local clone with the results downloaded from Drive), merge the rows and regenerate the website data and this page:

```bash
python benchmarks/throughput/merge_into_catalog.py \
    --all-rows /content/drive/MyDrive/olm-bench/t4/results/t4_context_all_batches_v2.csv \
    --selected-rows /content/drive/MyDrive/olm-bench/t4/results/t4_context_selected_v2.csv
python benchmarks/throughput/build_site_data.py
python benchmarks/throughput/render_docs_tables.py
```

Open a pull request with those changes and attach the suite's `results/`, `logs/`, `telemetry/` and `manifests/` directories as a zip so reviewers can audit the raw measurements. The same steps work for any GPU; only T4-class cards (compute capability below 8.0) need `AMP_DTYPE=fp16`.
