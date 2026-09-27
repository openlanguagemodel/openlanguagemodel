#!/usr/bin/env python3
"""Append a new GPU's exported rows to the canonical catalogs.

``run_publication_suite.sh`` ends by exporting every measured batch of a
suite to ``<prefix>_context_all_batches_v2.csv`` and the best batch per seed
to ``<prefix>_context_selected_v2.csv``, both in the catalog's column layout.
This script appends them to ``data/benchmark_catalog.csv`` and
``data/benchmark_catalog_selected.csv`` after checking that:

- the rows use the catalog's columns (``amp_dtype`` may be new: rows exported
  before ``--amp-dtype`` existed are all BF16 and are filled in as such);
- every row comes from one GPU board and both protocol seeds are present;
- no row repeats a (board, seed, model, context, batch) already in the file.

Rebuild the website data and the docs tables afterwards:

    python benchmarks/throughput/merge_into_catalog.py \\
        --all-rows RESULTS/t4_context_all_batches_v2.csv \\
        --selected-rows RESULTS/t4_context_selected_v2.csv
    python benchmarks/throughput/build_site_data.py
    python benchmarks/throughput/render_docs_tables.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from catalog import CATALOG_PATH, DATA_DIR, PROTOCOL_SEEDS

SELECTED_PATH = DATA_DIR / "benchmark_catalog_selected.csv"
ROW_KEY = ("gpu_uuid", "seed", "model_key", "sequence_length", "batch_size")
LEGACY_DEFAULTS = {"amp_dtype": "bf16"}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def merged_columns(existing: list[str], new: list[str], name: str) -> list[str]:
    """Catalog columns plus any new column that has a legacy default."""

    unknown = [column for column in new if column not in existing]
    unsupported = [column for column in unknown if column not in LEGACY_DEFAULTS]
    if unsupported:
        raise SystemExit(f"{name}: unknown columns {', '.join(unsupported)}")
    missing = [column for column in existing if column not in new]
    if missing:
        raise SystemExit(f"{name}: missing columns {', '.join(missing)}")
    return existing + unknown


def check_new_rows(rows: list[dict[str, str]], name: str) -> None:
    if not rows:
        raise SystemExit(f"{name}: no rows")
    boards = {row["gpu_uuid"] for row in rows}
    if len(boards) != 1:
        raise SystemExit(f"{name}: expected one GPU board, found {len(boards)}")
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(PROTOCOL_SEEDS):
        raise SystemExit(
            f"{name}: expected seeds {sorted(PROTOCOL_SEEDS)}, found {sorted(seeds)}"
        )


def merge(target: Path, source: Path) -> int:
    """Append ``source`` rows to ``target``; return the number appended."""

    existing_columns, existing_rows = read_rows(target)
    new_columns, new_rows = read_rows(source)
    check_new_rows(new_rows, source.name)
    columns = merged_columns(existing_columns, new_columns, source.name)

    seen = {tuple(row[field] for field in ROW_KEY) for row in existing_rows}
    for row in new_rows:
        key = tuple(row[field] for field in ROW_KEY)
        if key in seen:
            raise SystemExit(f"{source.name}: row already in {target.name}: {key}")
        seen.add(key)

    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({**LEGACY_DEFAULTS, **row})
        writer.writerows(new_rows)
    return len(new_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all-rows", type=Path, required=True)
    parser.add_argument("--selected-rows", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--selected-catalog", type=Path, default=SELECTED_PATH)
    args = parser.parse_args()

    appended = merge(args.catalog, args.all_rows)
    print(f"Appended {appended} rows to {args.catalog}")
    appended = merge(args.selected_catalog, args.selected_rows)
    print(f"Appended {appended} rows to {args.selected_catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
