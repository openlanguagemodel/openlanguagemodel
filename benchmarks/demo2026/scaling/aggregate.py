"""Aggregate weak-scaling JSON runs into efficiency summary."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List

from benchmarks.demo2026 import provenance


def load_runs(pattern: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_gpu: Dict[int, List[float]] = defaultdict(list)
    by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for run in runs:
        g = run["gpu_count"]
        by_gpu[g].append(run["tokens_per_second"])
        by_key[(g, run.get("replicate", 0))].append(run)

    baseline = statistics.mean(by_gpu.get(1, [0.0])) if by_gpu.get(1) else None

    rows = []
    for gpu_count in sorted(by_gpu.keys()):
        throughputs = by_gpu[gpu_count]
        mean_tps = statistics.mean(throughputs)
        std_tps = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        efficiency = (
            mean_tps / (gpu_count * baseline) if baseline and baseline > 0 else None
        )
        sample = next(r for r in runs if r["gpu_count"] == gpu_count)
        rows.append(
            {
                "gpu_count": gpu_count,
                "replicate_count": len(throughputs),
                "mean_tokens_per_second": mean_tps,
                "std_tokens_per_second": std_tps,
                "scaling_efficiency": efficiency,
                "mean_step_time_ms": sample.get("mean_step_time_ms"),
                "peak_memory_gb_per_gpu": sample.get("peak_memory_gb_per_gpu"),
            }
        )

    return {
        "framework": runs[0].get("framework", "olm") if runs else "olm",
        "baseline_1gpu_tokens_per_second": baseline,
        "rows": rows,
        "run_count": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/demo2026/results/raw/scaling")
    parser.add_argument(
        "--pattern", default="olm_*gpu_rep*.json"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/demo2026/results/derived/scaling_summary.json",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input, args.pattern)
    runs = load_runs(pattern)
    if not runs:
        raise SystemExit(f"No scaling runs found at {pattern}")

    summary = aggregate(runs)
    provenance.write_json(args.output, summary)

    md_path = args.output.replace(".json", ".md")
    lines = [
        "# Scaling summary",
        "",
        f"Baseline 1-GPU throughput: {summary.get('baseline_1gpu_tokens_per_second', 0):,.0f} tok/s",
        "",
        "| GPUs | mean tok/s | std | efficiency | peak mem (GB/GPU) |",
        "|------|------------|-----|------------|-------------------|",
    ]
    for row in summary["rows"]:
        eff = row["scaling_efficiency"]
        eff_s = f"{eff:.3f}" if eff is not None else "n/a"
        mem = row.get("peak_memory_gb_per_gpu") or [0]
        lines.append(
            f"| {row['gpu_count']} | {row['mean_tokens_per_second']:,.0f} | "
            f"{row['std_tokens_per_second']:,.0f} | {eff_s} | {mem[0]:.2f} |"
        )
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output} and {md_path}")


if __name__ == "__main__":
    main()
