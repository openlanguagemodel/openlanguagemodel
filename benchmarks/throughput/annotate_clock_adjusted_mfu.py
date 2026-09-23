#!/usr/bin/env python3
"""Add configured-clock MFU fields without altering raw benchmark JSON."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


GPU_REFERENCES = {
    "NVIDIA GeForce RTX 4090": {
        "nominal_dense_bf16_peak_tflops": 165.2,
        "official_reference_boost_clock_mhz": 2_520.0,
        "source": (
            "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/"
            "nvidia-ada-gpu-architecture.pdf"
        ),
    },
    "NVIDIA GeForce RTX 5090": {
        "nominal_dense_bf16_peak_tflops": 209.5,
        "official_reference_boost_clock_mhz": 2_407.0,
        "source": (
            "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
            "nvidia-rtx-blackwell-gpu-architecture.pdf"
        ),
    },
    "NVIDIA RTX A6000": {
        "nominal_dense_bf16_peak_tflops": 154.8,
        "official_reference_boost_clock_mhz": 1_800.0,
        "source": (
            "https://www.nvidia.com/content/PDF/"
            "nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf"
        ),
        "peak_derivation": (
            "NVIDIA specifies 154.8 TFLOP/s dense BF16 Tensor Core "
            "performance with FP32 accumulation. The separately listed "
            "309.6 TFLOP/s value uses structured sparsity."
        ),
    },
    "NVIDIA A100 80GB PCIe": {
        "nominal_dense_bf16_peak_tflops": 312.0,
        "official_reference_boost_clock_mhz": 1_410.0,
        "source": "https://www.nvidia.com/en-us/data-center/a100/",
        "clock_source": (
            "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
            "a100/pdf/PB-10577-001_v02.pdf"
        ),
        "peak_derivation": (
            "NVIDIA specifies 312 TFLOP/s dense BF16 Tensor Core "
            "performance. The separately listed 624 TFLOP/s value uses "
            "sparsity."
        ),
    },
    "NVIDIA H100 80GB HBM3": {
        "nominal_dense_bf16_peak_tflops": 989.5,
        "official_reference_boost_clock_mhz": 1_980.0,
        "source": "https://www.nvidia.com/en-us/data-center/h100/",
        "clock_source": (
            "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/"
            "performance.html"
        ),
        "peak_derivation": (
            "Half NVIDIA's official 1,979 TFLOP/s BF16 Tensor Core "
            "specification because that value uses sparsity."
        ),
    },
    "NVIDIA H200": {
        "nominal_dense_bf16_peak_tflops": 989.5,
        "official_reference_boost_clock_mhz": 1_980.0,
        "source": "https://www.nvidia.com/en-us/data-center/h200/",
        "clock_source": (
            "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/"
            "performance.html"
        ),
        "peak_derivation": (
            "Half NVIDIA's official 1,979 TFLOP/s H200 SXM BF16 Tensor "
            "Core specification because that value uses sparsity."
        ),
    },
    "NVIDIA B200": {
        "nominal_dense_bf16_peak_tflops": 2_250.0,
        "board_configured_reference_clock_mhz": 1_965.0,
        "clock_reference_kind": "measured_board_configured_maximum",
        "source": "https://www.nvidia.com/en-us/data-center/hgx/",
        "clock_source": (
            "benchmark environment: nvidia-smi "
            "--query-gpu=clocks.max.sm"
        ),
        "peak_derivation": (
            "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core "
            "performance for eight HGX B200 GPUs. Dense throughput is "
            "one-half of sparse, yielding 2,250 TFLOP/s per GPU."
        ),
    },
    "NVIDIA B300 SXM6 AC": {
        "nominal_dense_bf16_peak_tflops": 2_250.0,
        "board_configured_reference_clock_mhz": 2_032.0,
        "clock_reference_kind": "measured_board_configured_maximum",
        "source": "https://www.nvidia.com/en-us/data-center/hgx/",
        "clock_source": (
            "benchmark environment: nvidia-smi "
            "--query-gpu=clocks.max.sm"
        ),
        "peak_derivation": (
            "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core "
            "performance for eight HGX B300 GPUs. Dense throughput is "
            "one-half of sparse, yielding 2,250 TFLOP/s per GPU."
        ),
    },
    "NVIDIA B300-SXM6-AC": {
        "nominal_dense_bf16_peak_tflops": 2_250.0,
        "board_configured_reference_clock_mhz": 2_032.0,
        "clock_reference_kind": "measured_board_configured_maximum",
        "source": "https://www.nvidia.com/en-us/data-center/hgx/",
        "clock_source": (
            "benchmark environment: nvidia-smi "
            "--query-gpu=clocks.max.sm"
        ),
        "peak_derivation": (
            "NVIDIA specifies 36 PFLOP/s sparse BF16 Tensor Core "
            "performance for eight HGX B300 GPUs. Dense throughput is "
            "one-half of sparse, yielding 2,250 TFLOP/s per GPU."
        ),
    },
    "NVIDIA RTX 6000 Ada Generation": {
        "nominal_dense_bf16_peak_tflops": 364.2,
        "official_reference_boost_clock_mhz": 2_505.0,
        "source": (
            "https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/"
            "NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf"
        ),
        "peak_derivation": (
            "NVIDIA specifies 364.2 TFLOP/s dense BF16 Tensor Core "
            "performance with FP32 accumulation. The separately listed "
            "728.4 TFLOP/s value uses structured sparsity."
        ),
    },
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": {
        "nominal_dense_bf16_peak_tflops": 468.0,
        "official_reference_boost_clock_mhz": 2_430.0,
        "source": (
            "https://www.nvidia.com/en-us/data-center/"
            "rtx-pro-6000-blackwell-server-edition/"
        ),
        "peak_derivation": (
            "2 * NVIDIA's official 234 TFLOP/s dense TF32 specification. "
            "The advertised 1 PFLOP FP16/BF16 value uses sparsity."
        ),
    },
}


def configured_max_sm_clock(environment: dict[str, Any]) -> float:
    fields = [
        field.strip() for field in environment["nvidia_smi"].split(",")
    ]
    if len(fields) < 7:
        raise ValueError("nvidia_smi metadata does not contain max SM clock")
    return float(fields[6])


def registered_reference_clock(reference: dict[str, Any]) -> float:
    reference_clock = reference.get(
        "official_reference_boost_clock_mhz",
        reference.get("board_configured_reference_clock_mhz"),
    )
    if reference_clock is None:
        raise ValueError("GPU reference does not register a clock denominator")
    return float(reference_clock)


def annotate_node(node: Any, configured_peak: float) -> None:
    if isinstance(node, dict):
        if (
            "achieved_tflops" in node
            and "model_flops_utilization_pct" in node
        ):
            node["nominal_reference_model_flops_utilization_pct"] = node[
                "model_flops_utilization_pct"
            ]
            node["configured_clock_model_flops_utilization_pct"] = (
                100.0 * node["achieved_tflops"] / configured_peak
            )
        for value in node.values():
            annotate_node(value, configured_peak)
    elif isinstance(node, list):
        for value in node:
            annotate_node(value, configured_peak)


def annotate(record: dict[str, Any]) -> dict[str, Any]:
    annotated = copy.deepcopy(record)
    environment = annotated["environment"]
    device_name = environment["device_name"]
    if device_name not in GPU_REFERENCES:
        raise ValueError(f"no clock reference registered for {device_name!r}")
    reference = GPU_REFERENCES[device_name]
    nominal_peak = reference["nominal_dense_bf16_peak_tflops"]
    if environment["dense_bf16_peak_tflops"] != nominal_peak:
        raise ValueError("raw result does not use the registered nominal peak")

    configured_clock = configured_max_sm_clock(environment)
    reference_clock = registered_reference_clock(reference)
    configured_peak = (
        nominal_peak
        * configured_clock
        / reference_clock
    )
    raw_peak_source = environment.get("dense_bf16_peak_source")
    if raw_peak_source != reference["source"]:
        annotated.setdefault("metadata_corrections", []).append(
            {
                "field": "environment.dense_bf16_peak_source",
                "raw_value": raw_peak_source,
                "corrected_value": reference["source"],
                "reason": (
                    "The original harness used one shared source URL for "
                    "registered GPU peak values. The numerical peak and all "
                    "measurements are unchanged."
                ),
            }
        )
    environment.update(
        {
            "dense_bf16_peak_source": reference["source"],
            "configured_max_sm_clock_mhz": configured_clock,
            "configured_clock_dense_bf16_peak_tflops": configured_peak,
            "clock_reference_mhz": reference_clock,
            "clock_reference_kind": reference.get(
                "clock_reference_kind", "official_reference_boost"
            ),
            "clock_reference_source": reference.get(
                "clock_source", reference["source"]
            ),
        }
    )
    if "official_reference_boost_clock_mhz" in reference:
        environment["official_reference_boost_clock_mhz"] = reference_clock
    else:
        environment["board_configured_reference_clock_mhz"] = reference_clock
    annotated["mfu_reporting"] = {
        "raw_field": "model_flops_utilization_pct",
        "raw_field_denominator": "Official nominal dense BF16 peak.",
        "configured_clock_field": (
            "configured_clock_model_flops_utilization_pct"
        ),
        "configured_clock_denominator_tflops": configured_peak,
        "configured_clock_formula": (
            "nominal_peak * configured_max_sm_clock / reference_clock"
        ),
        "note": (
            "The nominal raw field is preserved. Configured-clock MFU is "
            "reported separately when the rental board's configured maximum "
            "clock differs from the reference clock."
        ),
        "peak_derivation": reference.get("peak_derivation"),
    }
    annotate_node(annotated, configured_peak)
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for input_path in args.inputs:
        output_path = (
            args.output_directory
            / f"{input_path.stem}_clock_adjusted.json"
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        record = annotate(json.loads(input_path.read_text()))
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_path)
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
