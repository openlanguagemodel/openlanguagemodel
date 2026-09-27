"""Tests for the single-GPU throughput catalog and everything generated from it.

The catalog scripts in ``benchmarks/throughput`` are standalone (they are not
part of the ``olm`` package), so this module puts that directory on the path.
Only the harness test at the bottom needs torch; it builds models on the meta
device, so no memory is allocated for weights.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPOSITORY_ROOT / "benchmarks" / "throughput"
sys.path.insert(0, str(BENCHMARK_DIR))

import build_site_data  # noqa: E402
import catalog  # noqa: E402
import lookup_training_point  # noqa: E402
import merge_into_catalog  # noqa: E402
import render_docs_tables  # noqa: E402


def _row(seed, batch, status="complete", tokens_per_second=None, stop_reason=""):
    return {
        "seed": seed,
        "batch": batch,
        "status": status,
        "stable": status == "complete",
        "tokens_per_second": tokens_per_second,
        "step_ms": None,
        "mfu_pct": None,
        "configured_mfu_pct": None,
        "allocated_gb": None,
        "allocated_pct": None,
        "reserved_gb": None,
        "stop_reason": stop_reason,
    }


def test_site_data_matches_the_catalog():
    text = build_site_data.serialize(build_site_data.build_payload())

    assert build_site_data.OUTPUT_PATH.read_text() == text, (
        "website/app/lib/benchmarks.generated.json is stale; run "
        "python benchmarks/throughput/build_site_data.py"
    )
    assert len(text.encode()) < build_site_data.SIZE_BUDGET_BYTES


def test_docs_tables_match_the_catalog():
    current = render_docs_tables.DOCS_PATH.read_text()
    rendered = render_docs_tables.render_page(
        current, render_docs_tables.render_blocks()
    )

    assert current == rendered, (
        "docs/benchmarks.md is stale; run "
        "python benchmarks/throughput/render_docs_tables.py"
    )


def test_cell_selects_fastest_batch_by_median_across_seeds():
    cell = catalog.summarize_cell(
        [
            _row(11, 4, tokens_per_second=100.0),
            _row(22, 4, tokens_per_second=110.0),
            _row(11, 8, tokens_per_second=120.0),
            _row(22, 8, tokens_per_second=118.0),
            _row(11, 16, "oom"),
            _row(22, 16, "oom"),
        ]
    )

    assert cell["status"] == "measured"
    assert cell["selected"]["batch"] == 8
    assert cell["selected"]["tokens_per_second"] == 119.0
    assert cell["first_oom_batch"] == 16
    assert cell["edge_batches"] == []
    assert not cell["search_capped"]


def test_cell_never_selects_a_batch_that_fit_only_one_seed():
    cell = catalog.summarize_cell(
        [
            _row(11, 8, tokens_per_second=100.0),
            _row(22, 8, tokens_per_second=100.0),
            _row(11, 16, tokens_per_second=150.0),
            _row(22, 16, "oom"),
        ]
    )

    assert cell["selected"]["batch"] == 8
    assert cell["edge_batches"] == [16]


def test_cell_without_a_complete_batch_is_oom():
    cell = catalog.summarize_cell([_row(11, 1, "oom"), _row(22, 1, "oom")])

    assert cell["status"] == "oom"
    assert cell["selected"] is None


def test_cell_flags_when_the_fastest_batch_is_the_largest_tried():
    capped = catalog.summarize_cell(
        [
            _row(11, 1, tokens_per_second=100.0),
            _row(22, 1, tokens_per_second=100.0),
            _row(11, 2, tokens_per_second=101.0),
            _row(22, 2, tokens_per_second=101.0),
        ]
    )
    saturated = catalog.summarize_cell(
        [
            _row(seed, batch, tokens_per_second=100.0 + batch, stop_reason=reason)
            for seed in (11, 22)
            for batch, reason in ((1, "saturation: no gain"), (2, "saturation"))
        ]
    )

    assert capped["search_capped"]
    assert not saturated["search_capped"]


def test_lookup_script_agrees_with_every_published_cell():
    cells = catalog.summarize_catalog(catalog.load_catalog())

    for (gpu, model, context), cell in cells.items():
        result = lookup_training_point.lookup(
            catalog.CATALOG_PATH, gpu, model, context, None
        )
        expected = cell["selected"]["batch"] if cell["selected"] else None
        assert result.get("best_batch_for_tokens_per_second") == expected


def test_every_measured_gpu_has_a_runpod_price_and_a_denominator():
    rows = catalog.load_catalog()
    runpod = catalog.load_price_profiles()["runpod"]

    for gpu, metadata in catalog.gpu_metadata(rows).items():
        assert gpu in runpod["gpus"]
        assert metadata["peak_tflops"] > 0
        assert metadata["configured_peak_tflops"] >= metadata["peak_tflops"]


def test_cost_per_billion_tokens():
    # 1e9 tokens at 1e5 tokens/s is 10,000 s, or 2.78 hours, at $1.80/h.
    assert catalog.usd_per_billion_tokens(1e5, 1.8) == pytest.approx(5.0)


def test_scaling_summary_matches_its_replicates():
    scaling = json.loads(catalog.SCALING_PATH.read_text())
    baseline = scaling["rows"][0]["mean_tokens_per_second"]

    for row in scaling["rows"]:
        replicates = row["replicate_tokens_per_second"]
        mean = sum(replicates) / len(replicates)
        assert row["mean_tokens_per_second"] == pytest.approx(mean)
        assert row["scaling_efficiency"] == pytest.approx(
            mean / (row["gpu_count"] * baseline)
        )


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _catalog_row(uuid, seed, batch):
    return {
        "gpu_uuid": uuid,
        "seed": str(seed),
        "model_key": "150m",
        "sequence_length": "2048",
        "batch_size": str(batch),
    }


def test_merge_appends_rows_and_backfills_amp_dtype(tmp_path):
    target = tmp_path / "catalog.csv"
    source = tmp_path / "t4.csv"
    columns = list(merge_into_catalog.ROW_KEY)
    _write_csv(target, columns, [_catalog_row("GPU-a", 11, 1)])
    _write_csv(
        source,
        columns + ["amp_dtype"],
        [
            {**_catalog_row("GPU-t4", seed, 1), "amp_dtype": "fp16"}
            for seed in catalog.PROTOCOL_SEEDS
        ],
    )

    assert merge_into_catalog.merge(target, source) == 2
    _, rows = merge_into_catalog.read_rows(target)
    assert [row["amp_dtype"] for row in rows] == ["bf16", "fp16", "fp16"]


def test_merge_rejects_duplicates_and_missing_seeds(tmp_path):
    target = tmp_path / "catalog.csv"
    columns = list(merge_into_catalog.ROW_KEY)
    _write_csv(
        target,
        columns,
        [_catalog_row("GPU-a", seed, 1) for seed in catalog.PROTOCOL_SEEDS],
    )
    duplicate = tmp_path / "duplicate.csv"
    _write_csv(
        duplicate,
        columns,
        [_catalog_row("GPU-a", seed, 1) for seed in catalog.PROTOCOL_SEEDS],
    )
    one_seed = tmp_path / "one_seed.csv"
    _write_csv(one_seed, columns, [_catalog_row("GPU-b", 11, 1)])

    with pytest.raises(SystemExit, match="already in"):
        merge_into_catalog.merge(target, duplicate)
    with pytest.raises(SystemExit, match="expected seeds"):
        merge_into_catalog.merge(target, one_seed)


def test_harness_models_match_catalog_parameter_counts():
    torch = pytest.importorskip("torch")
    harness = pytest.importorskip("benchmark_training_mfu")
    measured = catalog.model_parameters(catalog.load_catalog())

    for key, config in harness.MODEL_CONFIGS.items():
        with torch.device("meta"):
            model = harness.Llama3Model(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                intermediate_size=config.intermediate_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_seq_len=2_048,
                tie_weights=True,
            )
        assert harness.unique_parameter_count(model) == measured[key]["parameters"]


def test_harness_fp16_peak_and_loss_scaling():
    pytest.importorskip("torch")
    harness = pytest.importorskip("benchmark_training_mfu")

    peak, source, _ = harness.registered_peak("Tesla T4", "fp16")
    assert peak == 65.0
    assert source.startswith("https://www.nvidia.com/")
    assert harness.registered_peak("Tesla T4", "bf16")[0] is None
    assert harness.make_grad_scaler("bf16") is None
