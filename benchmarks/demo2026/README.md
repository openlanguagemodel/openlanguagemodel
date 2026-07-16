# OLM Demo 2026 Evidence Package

Reproducible numerical-correctness, breadth, and throughput evidence for the
AACL-IJCNLP 2026 OpenLanguageModel system demonstration.

## What this package claims

1. **Numerical correctness** — GPT-2, Llama 3, and Qwen 2.5 match independent
   Hugging Face Transformers implementations on logits, next-token loss, and
   selected gradients (tiny FP32 configs; dropout off; deterministic CPU).
2. **Breadth** — all documented **9 families / 27 named presets** pass
   construction, parameter-formula, tying, reduced forward-smoke, and
   checkpoint round-trip checks.
3. **Throughput / scaling** — weak scaling of a ~400M Llama-style model on
   1/2/4/8 H100s (single NVLink node), with an optional matched LitGPT baseline
   and an optional 7B FSDP execution smoke test.

Tolerances are **not** chosen in advance. Raw errors are recorded; regression
thresholds (if any) are derived after the first clean freeze.

## Environment

```bash
# From the repo root, with Python 3.10–3.12:
uv sync --extra dev --extra benchmark --python 3.12
# or: pip install -e ".[dev,benchmark]"
```

Reportable runs require a **clean git worktree**. Use `--allow-dirty` only for
local debugging (results are marked non-reportable).

Pin the environment used for reportable runs:

```bash
uv export --extra benchmark --frozen > benchmarks/demo2026/results/raw/environment.lock.txt
```

## Commands

### 1. Numerical parity

```bash
python -m benchmarks.demo2026.parity.run_parity \
  --family all --device cpu --seeds 11 22 33 \
  --output benchmarks/demo2026/results/raw/parity

pytest tests/hf_parity -q
```

### 2. Breadth (9 families / 27 presets)

```bash
python -m benchmarks.demo2026.breadth.run_breadth \
  --output benchmarks/demo2026/results/raw/breadth.json

pytest tests/test_model_breadth.py tests/test_model_smoke.py \
  tests/test_tied_embeddings.py tests/test_save_load.py -q
```

### 3. OLM scaling (H100 node)

Dry-run (1 GPU, short):

```bash
python -m benchmarks.demo2026.scaling.run_olm \
  --config benchmarks/demo2026/configs/scaling/llama400m.yaml \
  --gpu-count 1 --warmup-steps 5 --measured-steps 10 --allow-dirty
```

Full matrix:

```bash
bash benchmarks/demo2026/scaling/run_scaling.sh
python -m benchmarks.demo2026.scaling.aggregate \
  --input benchmarks/demo2026/results/raw/scaling \
  --output benchmarks/demo2026/results/derived/scaling_summary.json
```

### 4. LitGPT baseline (optional, 4-hour stop rule)

```bash
# Isolated env documented in litgpt/README.md
python -m benchmarks.demo2026.litgpt.validate_pair \
  --olm benchmarks/demo2026/results/raw/scaling \
  --litgpt benchmarks/demo2026/results/raw/litgpt
```

If a fair match cannot be established within four hours, keep diagnostic logs
and omit the cross-framework speed comparison from the paper.

### 5. FSDP smoke (optional, after core experiments)

```bash
torchrun --nproc_per_node=8 -m benchmarks.demo2026.fsdp_smoke \
  --steps 50 --output benchmarks/demo2026/results/raw/fsdp.json
```

### 6. Paper tables

```bash
python -m benchmarks.demo2026.report \
  --raw benchmarks/demo2026/results/raw \
  --out benchmarks/demo2026/results/derived
```

## Result layout

| Path | Contents |
|------|----------|
| `results/raw/` | Immutable per-run JSON (parity, breadth, scaling, fsdp) |
| `results/derived/` | Generated CSV/Markdown summaries (never hand-edit raw) |
| `schemas/` | JSON Schema for each result type |

## Interpretation rules

- Status `complete` means the run finished and metrics were recorded.
- Status `discrepancy` is reserved for post-freeze regression thresholds.
- Status `error` means the run crashed or produced non-finite values.
- If a bug fix changes model code, invalidate affected raw runs and re-run from
  the new frozen commit.
- Scaling efficiency is `throughput_N / (N × throughput_1)` (weak scaling).

## Hardware / runtime expectations

| Experiment | Hardware | Approx. wall time |
|------------|----------|-------------------|
| Parity (3 families × 3 seeds) | CPU | ~1–3 min |
| Breadth | CPU | ~2–5 min |
| Scaling 1/2/4/8 × 3 reps | 8×H100 NVLink | hours (node-dependent) |
| LitGPT 1+8 × 3 | same node | hours if attempted |
| FSDP 7B smoke | 8×H100 | tens of minutes |
