#!/usr/bin/env python3
"""Regenerate the measured tables in ``docs/benchmarks.md`` from the data files.

Every number on that page comes from this script. Each generated block sits
between a pair of HTML comments, which GitHub and the website both hide:

    <!-- BEGIN GENERATED: single-gpu-150m -->
    ...
    <!-- END GENERATED: single-gpu-150m -->

Everything between the markers is replaced; the prose around them is left
alone. ``--check`` exits non-zero when the page differs from a fresh render,
which is how the test suite keeps the docs and the CSV in agreement.

    python benchmarks/throughput/render_docs_tables.py
    python benchmarks/throughput/render_docs_tables.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from catalog import (
    SCALING_PATH,
    gpu_metadata,
    gpu_profile,
    gpus_by_price,
    load_catalog,
    load_price_profiles,
    model_parameters,
    summarize_catalog,
    usd_per_billion_tokens,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_PATH = REPOSITORY_ROOT / "docs" / "benchmarks.md"
HEADLINE_MODELS = ("150m", "500m")
HEADLINE_CONTEXT = 2_048

BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def day(timestamp: str) -> str:
    """``2026-07-24T03:17:33Z`` -> ``2026-07-24``."""

    return timestamp[:10]


def headline_table(
    model: str,
    cells: dict[tuple[str, str, int], dict[str, Any]],
    gpus: list[str],
    runpod: dict[str, Any],
) -> str:
    """One row per measured GPU for ``model`` at the headline context."""

    lines = [
        "| GPU | Best batch | Tokens/s | MFU (spec / board clock) | Peak VRAM "
        "| $ per 1B tokens, Community | $ per 1B tokens, Secure |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for gpu in gpus:
        profile = gpu_profile(gpu, runpod)
        name = profile["display_name"]
        if profile["vram_gb"] is not None and "GB" not in name:
            name += f" ({profile['vram_gb']} GB)"
        cell = cells.get((gpu, model, HEADLINE_CONTEXT))
        if cell is None or cell["selected"] is None:
            status = "out of memory" if cell else "not run"
            lines.append(f"| {name} | {status} | | | | | |")
            continue
        selected = cell["selected"]
        tokens_per_second = selected["tokens_per_second"]
        costs = []
        for tier in ("community_usd_per_hour", "secure_usd_per_hour"):
            rate = runpod["gpus"].get(gpu, {}).get(tier)
            costs.append(
                "no quote"
                if rate is None
                else f"${usd_per_billion_tokens(tokens_per_second, rate):.2f}"
            )
        lines.append(
            f"| {name} "
            f"| {selected['batch']} "
            f"| {tokens_per_second:,.0f} "
            f"| {selected['mfu_pct']:.1f}% / {selected['configured_mfu_pct']:.1f}% "
            f"| {selected['allocated_gb']:.1f} GB "
            f"| {costs[0]} "
            f"| {costs[1]} |"
        )
    return "\n".join(lines) + "\n"


def takeaways(
    cells: dict[tuple[str, str, int], dict[str, Any]],
    gpus: list[str],
    runpod: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> str:
    """Fastest GPU and cheapest GPU per RunPod tier for each headline model."""

    lines = []
    for model in HEADLINE_MODELS:
        measured = [
            (gpu, cells[(gpu, model, HEADLINE_CONTEXT)]["selected"])
            for gpu in gpus
            if (gpu, model, HEADLINE_CONTEXT) in cells
            and cells[(gpu, model, HEADLINE_CONTEXT)]["selected"] is not None
        ]
        fastest, fastest_cell = max(
            measured, key=lambda item: item[1]["tokens_per_second"]
        )
        cheapest = []
        for tier, label in (
            ("community_usd_per_hour", "Community"),
            ("secure_usd_per_hour", "Secure"),
        ):
            priced = [
                (
                    gpu,
                    usd_per_billion_tokens(
                        selected["tokens_per_second"], runpod["gpus"][gpu][tier]
                    ),
                )
                for gpu, selected in measured
                if gpu in runpod["gpus"]
            ]
            gpu, cost = min(priced, key=lambda item: item[1])
            cheapest.append(
                f"the {gpu_profile(gpu, runpod)['display_name']} on {label} "
                f"(${cost:.2f} per 1B tokens)"
            )
        lines.append(
            f"- **{models[model]['label']}:** the "
            f"{gpu_profile(fastest, runpod)['display_name']} is fastest "
            f"({fastest_cell['tokens_per_second']:,.0f} tokens/s). The cheapest "
            f"per token is {cheapest[0]} and {cheapest[1]}."
        )
    return "\n".join(lines) + "\n"


def scaling_table(scaling: dict[str, Any]) -> str:
    """Weak-scaling rows from the multi-GPU summary."""

    gpu = scaling["workload"]["gpu_model"].removeprefix("NVIDIA ")
    lines = [
        "| GPUs | Tokens/s (mean ± std) | Tokens/s per GPU | Scaling efficiency "
        "| Peak memory per GPU |",
        "|---|--:|--:|--:|--:|",
    ]
    for row in scaling["rows"]:
        count = row["gpu_count"]
        mean = row["mean_tokens_per_second"]
        lines.append(
            f"| {count} × {gpu} "
            f"| {mean:,.0f} ± {row['std_tokens_per_second']:,.0f} "
            f"| {mean / count:,.0f} "
            f"| {100.0 * row['scaling_efficiency']:.1f}% "
            f"| {max(row['peak_allocated_gb_per_gpu']):.2f} GB |"
        )
    return "\n".join(lines) + "\n"


def snapshot_note(
    rows: list[dict[str, Any]],
    gpus: list[str],
    runpod: dict[str, Any],
) -> str:
    """The one-paragraph provenance line printed above the headline tables."""

    models = model_parameters(rows)
    contexts = sorted({row["context"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    names = ", ".join(gpu_profile(gpu, runpod)["display_name"] for gpu in gpus)
    return (
        f"Catalog: {len(rows):,} rows covering {len(gpus)} GPUs ({names}), "
        f"{len(models)} model sizes from {next(iter(models.values()))['label']} "
        f"to {list(models.values())[-1]['label']}, contexts from "
        f"{contexts[0]:,} to {contexts[-1]:,} tokens and seeds "
        f"{' and '.join(str(seed) for seed in seeds)}. "
        f"Prices: RunPod on-demand rates captured on "
        f"{day(runpod['captured_at'])}.\n"
    )


def render_blocks() -> dict[str, str]:
    rows = load_catalog()
    cells = summarize_catalog(rows)
    runpod = load_price_profiles()["runpod"]
    gpus = gpus_by_price(gpu_metadata(rows), runpod)
    scaling = json.loads(SCALING_PATH.read_text())
    blocks = {
        "snapshot": snapshot_note(rows, gpus, runpod),
        "takeaways": takeaways(cells, gpus, runpod, model_parameters(rows)),
        "scaling": scaling_table(scaling),
    }
    for model in HEADLINE_MODELS:
        blocks[f"single-gpu-{model}"] = headline_table(model, cells, gpus, runpod)
    return blocks


def render_page(text: str, blocks: dict[str, str]) -> str:
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in blocks:
            raise KeyError(f"no generator for block {name!r}")
        seen.add(name)
        return match.group(1) + blocks[name] + match.group(4)

    rendered = BLOCK.sub(replace, text)
    missing = sorted(set(blocks) - seen)
    if missing:
        raise KeyError(f"{DOCS_PATH.name} has no markers for: {', '.join(missing)}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the page differs from a fresh render.",
    )
    args = parser.parse_args()

    current = args.docs.read_text()
    rendered = render_page(current, render_blocks())
    if args.check:
        if current != rendered:
            print(
                f"{args.docs} is stale; run {Path(__file__).name}",
                file=sys.stderr,
            )
            return 1
        print(f"OK {args.docs}")
        return 0
    args.docs.write_text(rendered)
    print(f"Wrote {args.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
