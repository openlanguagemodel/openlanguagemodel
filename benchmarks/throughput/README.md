# Single-GPU training throughput benchmark

This directory holds the harness that measures how fast OLM trains on one GPU, the catalog of every measurement it has produced, and the scripts that turn that catalog into the [GPU Benchmarks docs page](../../docs/benchmarks.md) and the website's `/benchmarks` explorer.

The harness lives outside `src/olm` on purpose: it depends on CUDA, `nvidia-smi` and rented hardware, and it is a measurement tool rather than library code. Everything else here (reading the catalog, building the site data, rendering the docs tables, merging contributions) needs only the Python standard library.

## Contents

| Path | What it does |
|---|---|
| `benchmark_training_mfu.py` | Batch-size sweep at 2,048 tokens for each model; writes one JSON per seed. |
| `benchmark_context_frontier.py` | Extends the sweep to other context lengths, starting from the 2,048-token result. |
| `run_publication_suite.sh` | Runs both seeds of both sweeps, the 16,384/32,768-token extensions, every validator and export, with GPU telemetry. |
| `run_with_gpu_telemetry.sh` | Runs one command while `nvidia-smi` samples clocks, power, temperature, utilization and memory every 200 ms. |
| `capture_environment_manifest.sh` | Records host, GPU, driver, Python packages and source revision (never the process environment). |
| `validate_training_results.py`, `validate_context_frontier.py`, `validate_gpu_telemetry.py` | Integrity checks: complete stages, matching replicate signatures, identical OOM frontiers across seeds, dense telemetry. |
| `audit_context_replicates.py` | Machine-readable cross-seed variation, kept even when the strict validator would reject a pair. |
| `annotate_clock_adjusted_mfu.py` | Adds board-clock MFU next to the raw MFU without rewriting the raw JSON. |
| `export_training_rows.py`, `export_context_frontier_rows.py` | Flatten raw JSON into the catalog's CSV layout. |
| `merge_into_catalog.py` | Appends a new GPU's exported rows to the catalogs after checking columns, seeds and duplicates. |
| `lookup_training_point.py` | Command-line lookup of one GPU/model/context point in the catalog. |
| `catalog.py` | Reads the catalog and applies the cell-selection rule used by every table. |
| `build_site_data.py` | Writes `website/app/lib/benchmarks.generated.json` for the website. |
| `render_docs_tables.py` | Regenerates the measured tables in `docs/benchmarks.md`. |
| `data/benchmark_catalog.csv` | The canonical, non-aggregated catalog: one row per GPU, seed, model, context and batch, including OOM attempts. |
| `data/benchmark_catalog_selected.csv` | Convenience view: the best batch per GPU, seed, model and context, or the smallest OOM batch when nothing fit. |
| `data/runpod_pricing_snapshot.json` | Dated RunPod Secure and Community on-demand rates for the GPUs on the host where the catalog was measured. |
| `data/provider_pricing_snapshot.json` | Dated quotes from Vast.ai, Lambda, AWS and Google Cloud, with the comparison policy. |
| `data/multi_gpu_scaling.json` | 1, 2 and 4 x A100 DDP weak-scaling runs from arXiv:2607.16669. |
| `data/rtx4090_metadata_erratum.json` | A source-URL correction for the RTX 4090 raw files; no measured value changed. |

Raw per-step JSON, telemetry traces, logs and manifests from the original runs are not in the repository; they are summarized, row for row, by the catalog.

## Protocol

### Models

Nine dense, tied-embedding Llama-3-style models built with `olm.models.meta.llama3.Llama3Model` (RoPE theta 500,000, no dropout, 32,000-token vocabulary). Parameter counts are measured on the built model, counting tied weights once.

| Key | Width | FFN width | Layers | Query heads | KV heads | Parameters |
|---|--:|--:|--:|--:|--:|--:|
| `150m` | 768 | 2,048 | 20 | 12 | 4 | 150,436,608 |
| `250m` | 896 | 2,432 | 26 | 14 | 2 | 246,397,312 |
| `350m` | 1,024 | 2,816 | 28 | 16 | 4 | 348,447,744 |
| `500m` | 1,280 | 3,456 | 27 | 20 | 4 | 505,516,800 |
| `700m` | 1,536 | 4,096 | 27 | 24 | 4 | 707,480,064 |
| `1b` | 1,792 | 4,864 | 28 | 28 | 4 | 995,135,232 |
| `2b` | 2,560 | 6,912 | 28 | 20 | 4 | 2,008,824,320 |
| `4b` | 3,584 | 9,728 | 29 | 28 | 4 | 3,999,611,392 |
| `8b` | 4,096 | 14,336 | 36 | 32 | 8 | 7,983,108,096 |

### The timed step

Each timed step is a forward pass under BF16 autocast, shifted-token cross-entropy, the backward pass and a fused AdamW update (`olm.train.optim.AdamW`, learning rate 1e-4, weight decay 0.1). Parameters, gradients and optimizer state stay in FP32. There is no activation checkpointing, no gradient accumulation and no `torch.compile`; attention is PyTorch SDPA with the backend PyTorch selects (recorded per row). Inputs are uniformly random token ids, generated once per batch configuration on the GPU and reused for every step, so no data loading is timed. Each step is timed with CUDA events.

With `--amp-dtype fp16` the step runs under FP16 autocast and adds dynamic loss scaling (`torch.amp.GradScaler`). This is for GPUs without BF16 Tensor Cores (compute capability below 8.0, such as the Tesla T4); BF16 runs on those GPUs are refused rather than timed on emulated BF16.

### Warmup and measurement

Each configuration warms up adaptively: after every step, the median of the last 5 step times is compared with the median of the 5 before. Warmup ends after 2 consecutive checks in which the medians differ by at most 2% and the robust jitter (1.4826 x MAD / median) is at most 3%, or after 50 steps. Then 20 steps are measured. Rows record whether warmup stabilized; every row in the current catalog did.

Tokens/s is `batch x context / median measured step time`. The mean-based value is kept alongside it.

### Batch and context sweeps

1. **2,048 tokens.** For each model the batch doubles from 1 through 64 and stops at the first out-of-memory (OOM) batch, or after two doublings in a row fail to add 1% throughput.
2. **512, 1,024, 4,096 and 8,192 tokens.** Starting from the fastest 2,048-token batch (the median over both seeds), the batch is scaled to keep tokens per step roughly constant and rounded to a power of two; that batch and its neighbors at half and double are tested.
3. **16,384 and 32,768 tokens.** The same rule, only for models that trained at the previous length in both seeds.

The whole protocol runs for seeds 11 and 22 on the same board. A model that does not fit at 2,048 tokens is not extended to other contexts, so those cells are *not run* rather than OOM.

### MFU

```text
training FLOPs per token = 6 N + 12 x layers x width x context
MFU = batch x context x FLOPs per token / step time / dense peak
```

`N` is the unique parameter count. The second term accounts for causal attention; elementwise operations are omitted. The dense peak is NVIDIA's published dense Tensor Core rate for the autocast dtype with FP32 accumulation. Where NVIDIA publishes only a sparse figure it is halved, and the RTX PRO 6000 peak is twice its published dense TF32 rate; sparse figures are never used directly.

The raw MFU uses that peak at NVIDIA's reference boost clock. Rented boards can run a higher maximum SM clock than the reference, so `annotate_clock_adjusted_mfu.py` also reports a board-clock MFU against the peak rescaled to the maximum SM clock that `nvidia-smi` reported. The catalog keeps both.

| GPU | Dense peak (TFLOP/s) | Reference clock (MHz) | Board-clock peak in the catalog |
|---|--:|--:|--:|
| RTX 4090 | 165.2 (BF16) | 2,520 | 205.517 |
| RTX 5090 | 209.5 (BF16) | 2,407 | 268.947 |
| A100 80GB PCIe | 312.0 (BF16) | 1,410 | 312.0 |
| RTX PRO 6000 Blackwell Server Edition | 468.0 (BF16) | 2,430 | 468.0 |
| H100 80GB HBM3 | 989.5 (BF16) | 1,980 | 989.5 |
| Tesla T4 | 65.0 (FP16) | 1,590 | not yet measured |

The source URL and derivation for every registered peak are in `benchmark_training_mfu.py` and `annotate_clock_adjusted_mfu.py`, and each raw result records the ones it used.

### Memory

Peak VRAM is `torch.cuda.max_memory_allocated()` over the whole batch (warmup and measured steps), reset before each batch; `max_memory_reserved()` is kept as well. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the suite runner.

## From rows to one number per cell

A *cell* is one GPU, model and context. `catalog.py` picks one batch per cell, and every consumer (the docs tables, the website, the tests) uses it:

1. keep complete, warmup-stabilized rows and group them by batch;
2. take the median of each metric across the seeds that completed that batch;
3. consider only batches that every seed with rows in the cell completed, so a batch that fit one seed at the edge of memory is never the headline, even when no other batch fit;
4. choose the batch with the highest median tokens/s (MFU picks the same batch, because for a fixed cell it is proportional to tokens/s).

A cell with rows but no batch that every seed completed is *OOM* when an attempt ran out of memory (every such cell in the current catalog) and *incomplete* otherwise. A cell with no rows was *not run*. When the fastest batch is also the largest batch the sweep tried, and it fit every seed, a larger batch might have been faster; those cells are flagged `search_capped`. Four RTX PRO 6000 cells for the 2B model have an edge-of-memory batch that fit seed 11 at about 98 GB but ran out of memory on seed 22; they are reported and never selected.

## Cost

```text
hours = tokens / tokens_per_second / 3,600
cost  = hours x USD per GPU-hour
```

Cost is steady-state compute on one GPU. It excludes setup and idle time, storage, data preparation, checkpointing, evaluation, failed runs and taxes. RunPod is the only host where performance was measured; costs for other providers apply the RunPod-measured tokens/s to their quoted rates, and some of those quotes are for a similar rather than identical product (the quote's `match_quality` says which). Every price has a capture date, and prices change.

## Run it

Look up a point in the catalog (no GPU needed):

```bash
python benchmarks/throughput/lookup_training_point.py \
    --gpu "RTX 5090" --model 500m --context 2048
```

Measure a GPU. From the repository root, on a machine with a CUDA GPU and OLM installed from this clone:

```bash
out=$HOME/olm-bench/my-gpu
benchmarks/throughput/capture_environment_manifest.sh \
    "$out/manifests/environment" "$(command -v python)" "$PWD"
benchmarks/throughput/run_publication_suite.sh \
    my_gpu "$out" "$(command -v python)" "$PWD"
```

Environment variables:

- `AMP_DTYPE=fp16` for GPUs without BF16 Tensor Cores.
- `MODELS=150m,250m` to limit the 2,048-token sweep and the main context sweep to some models. Published rows use all nine.

A GPU needs its dense peak and reference clock registered before it can be measured: add them, with NVIDIA's source, to `KNOWN_DENSE_BF16_TFLOPS` (or `KNOWN_DENSE_FP16_TFLOPS`) in `benchmark_training_mfu.py` and to `GPU_REFERENCES` in `annotate_clock_adjusted_mfu.py`.

The suite skips stages that already completed and stops, rather than overwrite, when it finds a stage that did not. `$out/results/` then holds a raw JSON per stage and seed, the clock-annotated copies, `<prefix>_context_all_batches_v2.csv` (every row, catalog layout) and `<prefix>_context_selected_v2.csv` (best batch per seed); `$out/logs/`, `$out/telemetry/` and `$out/manifests/` hold the rest, with a SHA-256 list of every file.

## Contribute a GPU

1. Run the suite as above. On a free Colab T4, follow the [Colab steps in the docs](../../docs/benchmarks.md#contribute-a-gpu).
2. Merge the rows and regenerate everything that depends on them:

   ```bash
   python benchmarks/throughput/merge_into_catalog.py \
       --all-rows "$out/results/my_gpu_context_all_batches_v2.csv" \
       --selected-rows "$out/results/my_gpu_context_selected_v2.csv"
   python benchmarks/throughput/build_site_data.py
   python benchmarks/throughput/render_docs_tables.py
   ```

3. Open a pull request with those changes, and attach `$out` (results, logs, telemetry and manifests) as a zip so reviewers can audit the raw data.

The catalog holds one board per GPU, and cells are keyed by GPU name, so `merge_into_catalog.py` refuses rows for a GPU that is already in it. It checks both files before writing either catalog.

If the GPU is rented somewhere other than RunPod, the website shows its measurements without a price until a dated quote is added to a price snapshot.

## Regenerate the published data

After any change under `data/` or to `catalog.py`:

```bash
python benchmarks/throughput/build_site_data.py
python benchmarks/throughput/render_docs_tables.py
```

`tests/test_benchmark_catalog.py` runs both with `--check`, so a stale website dataset or docs table fails the test suite.
