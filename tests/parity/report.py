"""Generate the reference-parity results table.

    PYTHONPATH=src:tests python -m parity.report           # markdown
    PYTHONPATH=src:tests python -m parity.report --json    # machine-readable

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from ._harness import environment, run_parity, set_determinism
from .architectures import ALL_CASES

DTYPES = [torch.float32, torch.float64]


def collect() -> list[dict]:
    rows = []
    for case in ALL_CASES:
        for dtype in DTYPES:
            rows.append(run_parity(case, dtype=dtype).as_dict())
    return rows


def to_markdown(rows: list[dict], env: dict[str, str]) -> str:
    out = ["# OLM reference-parity results", ""]
    out.append("| " + " | ".join(f"{k}: `{v}`" for k, v in env.items()) + " |")
    out.append("")
    scales = sorted({r["weight_scale"] for r in rows})
    out.append(
        "Reference matrix weights scaled by "
        + ", ".join(f"`x{s:g}`" for s in scales)
        + " before comparison; see `_harness.WEIGHT_SCALE`."
    )
    out.append("")
    # The relative columns come first because they are the reportable quantity:
    # the absolute differences move with the init scale, the config and the
    # platform, and are kept only so a reader can see the ulp-level size.
    out.append(
        "| architecture | reference | dtype | max abs dlogit / max abs logit | "
        "max abs dloss / loss | max abs logit diff | max abs loss diff | "
        "grad cosine sim | worst per-param cosine |"
    )
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| `{r['name']}` | {r['reference']} | {r['dtype']} | "
            f"{r['rel_logit_diff']:.3e} | {r['rel_loss_diff']:.3e} | "
            f"{r['max_abs_logit_diff']:.3e} | {r['max_abs_loss_diff']:.3e} | "
            f"{r['grad_cosine_sim']:.12f} | {r['min_per_tensor_grad_cosine']:.9f} |"
        )

    deviations = {
        r["name"]: r["reference_deviations"] for r in rows if r["reference_deviations"]
    }
    if deviations:
        out += ["", "## Reference configuration deviations", ""]
        for name, notes in deviations.items():
            for note in notes:
                out.append(f"- `{name}`: {note}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Pin threading before recording it, so the table reports the settings the
    # measurements were actually taken under.
    set_determinism()
    env = environment()
    rows = collect()
    print(json.dumps({"environment": env, "results": rows}, indent=2) if args.json
          else to_markdown(rows, env))
    return 0


if __name__ == "__main__":
    sys.exit(main())
