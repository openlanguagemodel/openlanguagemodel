"""Generate paper-ready tables from immutable raw JSON results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

from benchmarks.demo2026 import provenance


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def parity_table(raw_dir: str, out_dir: str) -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "parity", "*.json"))):
        r = load_json(path)
        rows.append(
            {
                "family": r["family"],
                "seed": r["input_seed"],
                "olm_commit": r.get("olm_commit"),
                "transformers": r.get("reference_library_version"),
                "params_olm": r.get("parameter_count_olm"),
                "params_ref": r.get("parameter_count_reference"),
                "max_logit_abs_err": r.get("max_logit_absolute_error"),
                "mean_logit_abs_err": r.get("mean_logit_absolute_error"),
                "loss_abs_err": r.get("loss_absolute_error"),
                "cos_emb": r.get("embedding_gradient_cosine"),
                "cos_early": r.get("early_layer_gradient_cosine"),
                "cos_late": r.get("late_layer_gradient_cosine"),
                "status": r.get("status"),
            }
        )
    csv_path = os.path.join(out_dir, "parity_results.csv")
    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_path = os.path.join(out_dir, "parity_results.md")
    lines = [
        "# Numerical parity (OLM vs Hugging Face Transformers)",
        "",
        "Tiny FP32 configs; dropout off; deterministic CPU. Tolerances were **not**",
        "chosen in advance — values below are measured errors.",
        "",
        "| family | seed | max |dlogit| | mean |dlogit| | |dloss| | cos(emb) | cos(early) | cos(late) | status |",
        "|--------|------|-------------|--------------|---------|----------|------------|-----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['seed']} | {r['max_logit_abs_err']:.3e} | "
            f"{r['mean_logit_abs_err']:.3e} | {r['loss_abs_err']:.3e} | "
            f"{r['cos_emb']:.6f} | {r['cos_early']:.6f} | {r['cos_late']:.6f} | {r['status']} |"
        )
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {csv_path} and {md_path}")


def breadth_summary(raw_dir: str, out_dir: str) -> None:
    path = os.path.join(raw_dir, "breadth.json")
    if not os.path.exists(path):
        print(f"Skip breadth: {path} not found")
        return
    data = load_json(path)
    summary = data.get("summary", {})
    md_path = os.path.join(out_dir, "breadth_summary.md")
    lines = [
        "# Breadth validation (9 families / 27 presets)",
        "",
        f"- Families checked: {summary.get('n_families', '?')}",
        f"- Presets checked: {summary.get('n_presets', '?')}",
        f"- Failed checks: {summary.get('n_failed_checks', '?')}",
        "",
        "See raw `breadth.json` for per-check evidence.",
    ]
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    provenance.write_json(os.path.join(out_dir, "breadth_summary.json"), summary)
    print(f"Wrote {md_path}")


def scaling_summary(raw_dir: str, out_dir: str) -> None:
    src = os.path.join(out_dir, "scaling_summary.json")
    if not os.path.exists(src):
        pattern = os.path.join(raw_dir, "scaling", "olm_*gpu_rep*.json")
        runs = [load_json(p) for p in sorted(glob.glob(pattern))]
        if not runs:
            print("Skip scaling: no runs")
            return
        from benchmarks.demo2026.scaling.aggregate import aggregate

        provenance.write_json(src, aggregate(runs))
    print(f"Scaling summary at {src}")


def environment_appendix(raw_dir: str, out_dir: str) -> None:
    # Use latest parity record for environment snapshot
    parity_files = sorted(glob.glob(os.path.join(raw_dir, "parity", "*.json")))
    env = load_json(parity_files[0])["environment"] if parity_files else {}
    path = os.path.join(out_dir, "environment.md")
    lines = ["# Environment / provenance", ""]
    for k, v in sorted(env.items()):
        if k == "env_vars":
            continue
        lines.append(f"- **{k}**: `{v}`")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="benchmarks/demo2026/results/raw")
    parser.add_argument("--out", default="benchmarks/demo2026/results/derived")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parity_table(args.raw, args.out)
    breadth_summary(args.raw, args.out)
    scaling_summary(args.raw, args.out)
    environment_appendix(args.raw, args.out)


if __name__ == "__main__":
    main()
