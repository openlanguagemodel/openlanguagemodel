#!/usr/bin/env python3
"""Validate timestamped nvidia-smi CSV traces from benchmark suites."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def validate(path: Path) -> tuple[int, float]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path}: empty telemetry file") from error
        if not header or header[0].strip() != "timestamp":
            raise ValueError(f"{path}: first column is not timestamp")
        if len(header) != len(set(header)):
            raise ValueError(f"{path}: duplicate columns")

        timestamps: list[datetime] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(header)} columns, "
                    f"found {len(row)}"
                )
            timestamp = datetime.strptime(
                row[0].strip(),
                TIMESTAMP_FORMAT,
            )
            if timestamps and timestamp < timestamps[-1]:
                raise ValueError(f"{path}:{line_number}: timestamp moved backwards")
            timestamps.append(timestamp)

    if not timestamps:
        raise ValueError(f"{path}: header exists but there are no samples")
    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    return len(timestamps), duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    total_samples = 0
    for path in args.inputs:
        samples, duration = validate(path)
        total_samples += samples
        print(f"VALID {path.name}: samples={samples}, " f"span={duration:.3f}s")
    print(f"VALID telemetry set: files={len(args.inputs)}, " f"samples={total_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
