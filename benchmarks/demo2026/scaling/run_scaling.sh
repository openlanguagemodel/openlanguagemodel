#!/usr/bin/env bash
# Weak-scaling matrix: 1/2/4/8 GPUs × 3 replicates.
# Requires a single NVLink node with at least 8 GPUs for the full matrix.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CONFIG="benchmarks/demo2026/configs/scaling/llama400m.yaml"
OUT="benchmarks/demo2026/results/raw/scaling"
mkdir -p "$OUT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

for REP in 0 1 2; do
  for N in 1 2 4 8; do
  echo "=== OLM scaling: ${N} GPU(s), replicate ${REP} ==="
  if [[ "$N" -eq 1 ]]; then
    "$PYTHON" -m benchmarks.demo2026.scaling.run_olm \
      --config "$CONFIG" \
      --gpu-count 1 \
      --replicate "$REP" \
      --output "$OUT"
  else
    torchrun --nproc_per_node="$N" -m benchmarks.demo2026.scaling.run_olm \
      --config "$CONFIG" \
      --gpu-count "$N" \
      --replicate "$REP" \
      --output "$OUT"
  fi
  done
done

echo "Scaling runs complete. Aggregate with:"
echo "  $PYTHON -m benchmarks.demo2026.scaling.aggregate --input $OUT --output benchmarks/demo2026/results/derived/scaling_summary.json"
