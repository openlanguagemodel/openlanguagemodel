#!/usr/bin/env bash
#
# Run the two-seed OLM training batch sweep and context-length frontier.
#
# The runner is intentionally GPU-agnostic. The benchmark's registered
# nominal dense-BF16 denominator is used unless PEAK_TFLOPS is supplied.
# Completed stages are skipped; an incomplete JSON is preserved and causes a
# hard stop so a partial result is never silently overwritten.
#
# Optional environment variables:
#   AMP_DTYPE  bf16 (default) or fp16 for GPUs without BF16 Tensor Cores (T4).
#   MODELS     comma-separated model keys for the 2,048-token sweep and main
#              frontier (default: all nine). Publication rows use all nine.

set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo \
        "usage: $0 PREFIX OUTPUT_ROOT PYTHON_EXECUTABLE REPOSITORY_ROOT [PEAK_TFLOPS]" \
        >&2
    exit 2
fi

prefix=$1
output_root=$2
python_executable=$3
repository_root=$4
peak_tflops=${5:-}
amp_dtype=${AMP_DTYPE:-bf16}
models=${MODELS:-all}

scripts_dir="$repository_root/benchmarks/throughput"
results_dir="$output_root/results"
logs_dir="$output_root/logs"
telemetry_dir="$output_root/telemetry"
manifest_dir="$output_root/manifests/publication"
mkdir -p "$results_dir" "$logs_dir" "$telemetry_dir" "$manifest_dir"

export PYTHONPATH="$repository_root/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repository_root"

peak_arguments=()
if [[ -n "$peak_tflops" ]]; then
    peak_arguments=(--peak-tflops "$peak_tflops")
fi

is_complete_json() {
    local path=$1
    [[ -f "$path" ]] || return 1
    "$python_executable" - "$path" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    record = json.load(handle)
raise SystemExit(0 if record.get("status") == "complete" else 1)
PY
}

run_benchmark() {
    local output_path=$1
    local stage_name=$2
    shift 2

    if is_complete_json "$output_path"; then
        echo "SKIP complete stage: $stage_name"
        return
    fi
    if [[ -e "$output_path" ]]; then
        echo "ERROR incomplete stage is preserved: $output_path" >&2
        exit 1
    fi

    "$scripts_dir/run_with_gpu_telemetry.sh" \
        "$telemetry_dir/${stage_name}.csv" \
        "$logs_dir/${stage_name}.log" \
        "$@"
}

select_common_survivors() {
    local context_length=$1
    shift
    "$python_executable" - "$context_length" "$@" <<'PY'
import json
import sys

model_order = ["150m", "250m", "350m", "500m", "700m", "1b", "2b", "4b", "8b"]
context_length = int(sys.argv[1])
records = [json.load(open(path)) for path in sys.argv[2:]]

def completed(record, key):
    model = next((item for item in record["models"] if item["key"] == key), None)
    if model is None:
        return False
    context = next(
        (
            item
            for item in model.get("contexts", [])
            if item.get("sequence_length") == context_length
        ),
        None,
    )
    return (
        context is not None
        and context.get("status") == "complete"
        and any(
            batch.get("status") == "complete"
            for batch in context.get("batches", [])
        )
    )

survivors = [
    key for key in model_order if all(completed(record, key) for record in records)
]
print(",".join(survivors))
PY
}

training_files=()
for seed in 11 22; do
    stage="${prefix}_training_seed${seed}_v2"
    output="$results_dir/${stage}.json"
    run_benchmark \
        "$output" \
        "$stage" \
        "$python_executable" \
        "$scripts_dir/benchmark_training_mfu.py" \
        --output "$output" \
        --seed "$seed" \
        --amp-dtype "$amp_dtype" \
        --models "$models" \
        "${peak_arguments[@]}"
    training_files+=("$output")
done

"$python_executable" "$scripts_dir/validate_training_results.py" \
    "${training_files[@]}" \
    | tee "$logs_dir/${prefix}_training_validation_v2.log"

context_files=()
for seed in 11 22; do
    stage="${prefix}_context_frontier_seed${seed}_v2"
    output="$results_dir/${stage}.json"
    baseline="$results_dir/${prefix}_training_seed${seed}_v2.json"
    run_benchmark \
        "$output" \
        "$stage" \
        "$python_executable" \
        "$scripts_dir/benchmark_context_frontier.py" \
        --baseline "$baseline" \
        --candidate-baselines "${training_files[@]}" \
        --output "$output" \
        --seed "$seed" \
        --models "$models" \
        "${peak_arguments[@]}"
    context_files+=("$output")
done

"$python_executable" "$scripts_dir/validate_context_frontier.py" \
    "${context_files[@]}" \
    | tee "$logs_dir/${prefix}_context_frontier_validation_v2.log"
"$python_executable" "$scripts_dir/audit_context_replicates.py" \
    --output "$results_dir/${prefix}_context_frontier_replicate_audit_v2.json" \
    "${context_files[@]}"

models_16k=$(select_common_survivors 8192 "${context_files[@]}")
extension_16k_files=()
if [[ -n "$models_16k" ]]; then
    for seed in 11 22; do
        stage="${prefix}_context_extension_16384_seed${seed}_v2"
        output="$results_dir/${stage}.json"
        baseline="$results_dir/${prefix}_training_seed${seed}_v2.json"
        run_benchmark \
            "$output" \
            "$stage" \
            "$python_executable" \
            "$scripts_dir/benchmark_context_frontier.py" \
            --baseline "$baseline" \
            --candidate-baselines "${training_files[@]}" \
            --output "$output" \
            --seed "$seed" \
            --models "$models_16k" \
            --contexts 16384 \
            "${peak_arguments[@]}"
        extension_16k_files+=("$output")
    done
    "$python_executable" "$scripts_dir/validate_context_frontier.py" \
        --allow-all-oom \
        "${extension_16k_files[@]}" \
        | tee "$logs_dir/${prefix}_context_extension_16384_validation_v2.log"
    "$python_executable" "$scripts_dir/audit_context_replicates.py" \
        --output \
        "$results_dir/${prefix}_context_extension_16384_replicate_audit_v2.json" \
        "${extension_16k_files[@]}"
fi

extension_32k_files=()
if [[ ${#extension_16k_files[@]} -gt 0 ]]; then
    models_32k=$(select_common_survivors 16384 "${extension_16k_files[@]}")
    if [[ -n "$models_32k" ]]; then
        for seed in 11 22; do
            stage="${prefix}_context_extension_32768_seed${seed}_v2"
            output="$results_dir/${stage}.json"
            baseline="$results_dir/${prefix}_training_seed${seed}_v2.json"
            run_benchmark \
                "$output" \
                "$stage" \
                "$python_executable" \
                "$scripts_dir/benchmark_context_frontier.py" \
                --baseline "$baseline" \
                --candidate-baselines "${training_files[@]}" \
                --output "$output" \
                --seed "$seed" \
                --models "$models_32k" \
                --contexts 32768 \
                "${peak_arguments[@]}"
            extension_32k_files+=("$output")
        done
        "$python_executable" "$scripts_dir/validate_context_frontier.py" \
            --allow-all-oom \
            "${extension_32k_files[@]}" \
            | tee "$logs_dir/${prefix}_context_extension_32768_validation_v2.log"
        "$python_executable" "$scripts_dir/audit_context_replicates.py" \
            --output \
            "$results_dir/${prefix}_context_extension_32768_replicate_audit_v2.json" \
            "${extension_32k_files[@]}"
    fi
fi

publication_files=(
    "${training_files[@]}"
    "${context_files[@]}"
    "${extension_16k_files[@]}"
    "${extension_32k_files[@]}"
)

"$python_executable" "$scripts_dir/annotate_clock_adjusted_mfu.py" \
    --output-directory "$results_dir/clock-adjusted" \
    "${publication_files[@]}"
"$python_executable" "$scripts_dir/export_training_rows.py" \
    --output "$results_dir/${prefix}_training_all_batches_v2.csv" \
    "${training_files[@]}"
"$python_executable" "$scripts_dir/export_context_frontier_rows.py" \
    --all-output "$results_dir/${prefix}_context_all_batches_v2.csv" \
    --selected-output "$results_dir/${prefix}_context_selected_v2.csv" \
    "${context_files[@]}" \
    "${extension_16k_files[@]}" \
    "${extension_32k_files[@]}"

telemetry_files=("$telemetry_dir/${prefix}"_*.csv)
"$python_executable" "$scripts_dir/validate_gpu_telemetry.py" \
    "${telemetry_files[@]}" \
    | tee "$logs_dir/${prefix}_telemetry_validation_v2.log"

nvidia-smi -q >"$manifest_dir/nvidia_smi_post_run.txt"
find "$results_dir" "$logs_dir" "$telemetry_dir" "$manifest_dir" \
    -type f \
    ! -name artifact_sha256.txt \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$manifest_dir/artifact_sha256.txt"
touch "$output_root/PUBLICATION_SUITE_COMPLETE"

echo "COMPLETE publication suite: $prefix"
