# LitGPT baseline adapter (optional)

LitGPT is **not** a runtime dependency of OLM. Install in an isolated environment
when attempting a fair cross-framework throughput comparison.

## Pin (example)

```bash
python -m venv .venv-litgpt
source .venv-litgpt/bin/activate
pip install 'litgpt>=0.5.0' torch pyyaml
```

Record the exact LitGPT commit/tag in every result JSON (`environment.litgpt_version`).

## Fairness checklist

Before comparing throughput, run:

```bash
python -m benchmarks.demo2026.litgpt.validate_pair \
  --olm-config benchmarks/demo2026/configs/scaling/llama400m.yaml \
  --litgpt-config benchmarks/demo2026/litgpt/litgpt_llama400m.yaml
```

The validator refuses comparison when any of these differ:

- vocabulary size, block size, layers, hidden size, heads, query groups
- intermediate / MLP size, parameter count (unique)
- precision, local batch, grad accumulation, optimizer fused flag
- attention backend selection, `torch.compile` flag
- synthetic ring seed and staging

## Runs (1 GPU and 6 GPU only)

```bash
# 1 GPU
python -m benchmarks.demo2026.litgpt.run_litgpt \
  --config benchmarks/demo2026/litgpt/litgpt_llama400m.yaml \
  --gpu-count 1 --replicate 0 \
  --output benchmarks/demo2026/results/raw/litgpt

# 6 GPU
torchrun --nproc_per_node=6 -m benchmarks.demo2026.litgpt.run_litgpt \
  --config benchmarks/demo2026/litgpt/litgpt_llama400m.yaml \
  --gpu-count 6 --replicate 0 \
  --output benchmarks/demo2026/results/raw/litgpt
```

## Four-hour stop rule

If a matched LitGPT setup cannot be validated and run within **four hours** of
debugging, keep diagnostic logs but **omit** the cross-framework speed comparison
from the paper. Report OLM scaling and memory only.
