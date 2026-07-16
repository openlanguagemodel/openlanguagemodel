"""LitGPT throughput adapter — same measurement protocol as OLM scaling.

Requires ``litgpt`` in the active environment. If import fails, exits with a
clear message (four-hour stop rule: omit cross-framework comparison).
"""

from __future__ import annotations

import argparse
import importlib
import sys

from benchmarks.demo2026.scaling.run_olm import main as olm_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--output", default="benchmarks/demo2026/results/raw/litgpt")
    parser.add_argument("--allow-dirty", action="store_true")
    args, _ = parser.parse_known_args()

    try:
        importlib.import_module("litgpt")
    except ImportError:
        print(
            "LitGPT is not installed. See benchmarks/demo2026/litgpt/README.md. "
            "Per the four-hour stop rule, omit cross-framework speed comparison.",
            file=sys.stderr,
        )
        return 2

    # LitGPT ships its own training entrypoints; for a fair apples-to-apples
    # measurement we delegate to the OLM harness protocol documentation and
    # record that operators must wire LitGPT's pretrain script with matched
    # hyperparameters from litgpt_llama400m.yaml. A full in-tree LitGPT model
    # wrapper is intentionally out of scope for the library itself.
    print(
        "LitGPT is installed. Run LitGPT pretrain with matched config from "
        "benchmarks/demo2026/litgpt/litgpt_llama400m.yaml and convert logs to "
        "benchmarks/demo2026/schemas/scaling_result.schema.json manually, or "
        "extend this adapter with a LitGPT Config -> training loop bridge.",
        file=sys.stderr,
    )
    print(
        "For now, use validate_pair.py to confirm static fairness, then run "
        "LitGPT's official pretrain recipe on the A100 SXM node.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
