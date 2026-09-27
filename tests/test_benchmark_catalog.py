"""Tests for the single-GPU throughput catalog and everything generated from it.

The catalog scripts in ``benchmarks/throughput`` are standalone (they are not
part of the ``olm`` package), so this module puts that directory on the path.
Only the harness test at the bottom needs torch; it builds models on the meta
device, so no memory is allocated for weights.
"""

import csv
import json
import sys
import warnings
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


def test_site_data_displays_the_same_digits_as_the_docs():
    # The page formats the stored (rounded) values; the docs tables format the
    # raw ones. Both must print the same digits.
    payload = build_site_data.build_payload()
    cells = catalog.summarize_catalog(catalog.load_catalog())
    shown = {
        "tps": ("tokens_per_second", "{:.0f}"),
        "mfu": ("mfu_pct", "{:.1f}"),
        "mfuBoard": ("configured_mfu_pct", "{:.1f}"),
        "vramGb": ("allocated_gb", "{:.1f}"),
    }

    for cell in payload["cells"]:
        raw = cells[(cell["gpu"], cell["model"], cell["context"])]["batches"]
        for stored, original in zip(cell["batches"], raw):
            for key, (field, spec) in shown.items():
                assert spec.format(stored[key]) == spec.format(original[field])

    scaling = json.loads(catalog.SCALING_PATH.read_text())["rows"]
    for stored, original in zip(payload["scaling"]["rows"], scaling):
        mean = original["mean_tokens_per_second"]
        assert f"{stored['tps']:.0f}" == f"{mean:.0f}"
        assert f"{stored['tps'] / stored['gpus']:.0f}" == f"{mean / stored['gpus']:.0f}"
        assert f"{stored['tpsStd']:.0f}" == f"{original['std_tokens_per_second']:.0f}"
        assert f"{100 * stored['efficiency']:.1f}" == (
            f"{100 * original['scaling_efficiency']:.1f}"
        )


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
    # Seed 22 ran out of memory at every batch, so the batch that fit seed 11
    # is the only complete one and must still not be selected.
    edge_only = catalog.summarize_cell(
        [
            _row(11, 1, tokens_per_second=100.0),
            _row(11, 2, "oom"),
            _row(22, 1, "oom"),
        ]
    )

    assert cell["selected"]["batch"] == 8
    assert cell["edge_batches"] == [16]
    assert edge_only["status"] == "oom"
    assert edge_only["selected"] is None
    assert edge_only["edge_batches"] == [1]
    assert not edge_only["search_capped"]


def test_cell_without_a_complete_batch_is_oom():
    cell = catalog.summarize_cell([_row(11, 1, "oom"), _row(22, 1, "oom")])

    assert cell["status"] == "oom"
    assert cell["selected"] is None


def test_cell_where_a_seed_failed_without_oom_is_incomplete():
    cell = catalog.summarize_cell(
        [_row(11, 1, tokens_per_second=100.0), _row(22, 1, "error")]
    )

    assert cell["status"] == "incomplete"
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


def test_lookup_script_never_selects_an_edge_of_memory_batch(tmp_path):
    path = tmp_path / "catalog.csv"
    rows = [
        {"seed": "11", "batch_size": "1", "status": "complete", "stable": "True"},
        {"seed": "11", "batch_size": "2", "status": "oom", "stable": ""},
        {"seed": "22", "batch_size": "1", "status": "oom", "stable": ""},
    ]
    _write_csv(
        path,
        ["gpu_name", "gpu_uuid", "model_key", "sequence_length", *rows[0]],
        [
            {
                "gpu_name": "Tesla T4",
                "gpu_uuid": "GPU-t4",
                "model_key": "150m",
                "sequence_length": "2048",
                **row,
            }
            for row in rows
        ],
    )

    result = lookup_training_point.lookup(path, "T4", "150m", 2048, None)

    assert result["status"] == "no_batch_fit_every_seed"
    assert result["edge_of_memory_batches"] == [1]
    assert "best_batch_for_tokens_per_second" not in result


def test_every_measured_gpu_has_an_mfu_denominator():
    # A measured GPU needs no RunPod price: GPUs rented elsewhere, like a
    # contributed Colab T4, are shown without a quote.
    rows = catalog.load_catalog()

    for metadata in catalog.gpu_metadata(rows).values():
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


def _catalog_row(uuid, seed, batch, gpu_name=None):
    return {
        "gpu_name": gpu_name or f"NVIDIA {uuid}",
        "gpu_uuid": uuid,
        "seed": str(seed),
        "model_key": "150m",
        "sequence_length": "2048",
        "batch_size": str(batch),
    }


def test_merge_appends_rows_and_backfills_amp_dtype(tmp_path):
    target = tmp_path / "catalog.csv"
    source = tmp_path / "t4.csv"
    columns = ["gpu_name", *merge_into_catalog.ROW_KEY]
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


def test_merge_rejects_duplicates_missing_seeds_and_a_second_board(tmp_path):
    target = tmp_path / "catalog.csv"
    columns = ["gpu_name", *merge_into_catalog.ROW_KEY]
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
    # Same GPU model on another board: its cells would mix with GPU-a's.
    second_board = tmp_path / "second_board.csv"
    _write_csv(
        second_board,
        columns,
        [
            _catalog_row("GPU-c", seed, 2, gpu_name="NVIDIA GPU-a")
            for seed in catalog.PROTOCOL_SEEDS
        ],
    )

    with pytest.raises(SystemExit, match="already in"):
        merge_into_catalog.merge(target, duplicate)
    with pytest.raises(SystemExit, match="expected seeds"):
        merge_into_catalog.merge(target, one_seed)
    with pytest.raises(SystemExit, match="one board per GPU"):
        merge_into_catalog.merge(target, second_board)


def _as_t4(row):
    """A real catalog row relabelled as an FP16 Tesla T4 row (synthetic)."""

    return {
        **row,
        "gpu_name": "Tesla T4",
        "gpu_uuid": "GPU-t4-test",
        "amp_dtype": "fp16",
        "dense_bf16_peak_tflops": "65.0",
        "configured_clock_dense_bf16_peak_tflops": "65.0",
    }


def test_contributed_t4_rows_flow_through_merge_site_and_docs(tmp_path, monkeypatch):
    # The documented contribution path: merge a T4 suite's exports, then
    # rebuild the site data and the docs tables. The T4 has no RunPod price.
    # Real T4 rows, once merged, are dropped from the copy so this keeps
    # exercising the merge.
    targets = {}
    for name, path in (
        ("all", catalog.CATALOG_PATH),
        ("selected", merge_into_catalog.SELECTED_PATH),
    ):
        columns, rows = merge_into_catalog.read_rows(path)
        rows = [row for row in rows if row["gpu_name"] != "Tesla T4"]
        targets[name] = tmp_path / path.name
        _write_csv(targets[name], columns, rows)
        t4_rows = [
            _as_t4(row)
            for row in rows
            if row["gpu_name"] == "NVIDIA GeForce RTX 4090"
            and row["model_key"] == "150m"
            and row["sequence_length"] == "2048"
        ]
        _write_csv(tmp_path / f"t4_{name}.csv", [*columns, "amp_dtype"], t4_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_into_catalog.py",
            "--all-rows",
            str(tmp_path / "t4_all.csv"),
            "--selected-rows",
            str(tmp_path / "t4_selected.csv"),
            "--catalog",
            str(targets["all"]),
            "--selected-catalog",
            str(targets["selected"]),
        ],
    )
    assert merge_into_catalog.main() == 0

    def merged_catalog():
        return catalog.load_catalog(targets["all"])

    monkeypatch.setattr(build_site_data, "load_catalog", merged_catalog)
    monkeypatch.setattr(render_docs_tables, "load_catalog", merged_catalog)
    payload = build_site_data.build_payload()
    blocks = render_docs_tables.render_blocks()

    t4 = next(gpu for gpu in payload["gpus"] if gpu["id"] == "Tesla T4")
    assert (t4["name"], t4["vramGb"], t4["ampDtype"]) == ("T4", 16, "fp16")
    assert "Tesla T4" not in {gpu["id"] for gpu in payload["pendingGpus"]}
    assert len(build_site_data.serialize(payload)) < build_site_data.SIZE_BUDGET_BYTES
    t4_line = next(
        line
        for line in blocks["single-gpu-150m"].splitlines()
        if line.startswith("| T4 (16 GB) |")
    )
    assert t4_line.endswith("| no quote | no quote |")
    assert "not run" not in t4_line and "out of memory" not in t4_line
    assert "| T4 (16 GB) | not run |" in blocks["single-gpu-500m"]
    render_docs_tables.render_page(render_docs_tables.DOCS_PATH.read_text(), blocks)


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
    torch = pytest.importorskip("torch")
    harness = pytest.importorskip("benchmark_training_mfu")

    peak, source, _ = harness.registered_peak("Tesla T4", "fp16")
    assert peak == 65.0
    assert source.startswith("https://www.nvidia.com/")
    assert harness.registered_peak("Tesla T4", "bf16")[0] is None
    assert harness.make_grad_scaler("bf16") is None
    with warnings.catch_warnings():
        # Without CUDA, GradScaler warns and disables itself; only the type
        # matters here.
        warnings.simplefilter("ignore", UserWarning)
        assert isinstance(harness.make_grad_scaler("fp16"), torch.amp.GradScaler)
