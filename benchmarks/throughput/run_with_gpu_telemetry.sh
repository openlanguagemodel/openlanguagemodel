#!/usr/bin/env bash
#
# Run one benchmark command while retaining a dense, timestamped GPU telemetry
# trace and the command's complete stdout/stderr stream.

set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 TELEMETRY_CSV COMMAND_LOG COMMAND [ARG ...]" >&2
    exit 2
fi

telemetry_csv=$1
command_log=$2
shift 2

mkdir -p "$(dirname "$telemetry_csv")" "$(dirname "$command_log")"

telemetry_fields=(
    timestamp
    name
    uuid
    pci.bus_id
    driver_version
    pstate
    utilization.gpu
    utilization.memory
    utilization.encoder
    utilization.decoder
    memory.total
    memory.used
    memory.free
    temperature.gpu
    fan.speed
    power.draw
    power.limit
    clocks.current.graphics
    clocks.current.sm
    clocks.current.memory
    clocks.current.video
    clocks.max.graphics
    clocks.max.sm
    clocks.max.memory
    pcie.link.gen.current
    pcie.link.gen.max
    pcie.link.width.current
    pcie.link.width.max
)
telemetry_query=$(IFS=,; echo "${telemetry_fields[*]}")

nvidia-smi \
    --query-gpu="$telemetry_query" \
    --format=csv,nounits \
    --loop-ms=200 \
    >"$telemetry_csv" \
    2>"${telemetry_csv%.csv}.stderr.log" &
telemetry_pid=$!

cleanup() {
    if kill -0 "$telemetry_pid" 2>/dev/null; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM HUP

{
    echo "wrapper_started_at=$(date --utc --iso-8601=ns)"
    printf "command="
    printf "%q " "$@"
    echo
    "$@"
    command_status=$?
    echo "wrapper_finished_at=$(date --utc --iso-8601=ns)"
    echo "command_exit_status=$command_status"
    exit "$command_status"
} 2>&1 | tee "$command_log"
command_status=${PIPESTATUS[0]}

exit "$command_status"
