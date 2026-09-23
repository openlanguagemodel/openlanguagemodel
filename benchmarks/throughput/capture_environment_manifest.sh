#!/usr/bin/env bash
#
# Capture a reproducibility-oriented host, GPU, Python, package, and source
# manifest without dumping the process environment (which may contain secrets).

set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 OUTPUT_DIR PYTHON_EXECUTABLE REPOSITORY_ROOT" >&2
    exit 2
fi

output_dir=$1
python_executable=$2
repository_root=$3
mkdir -p "$output_dir"

run_capture() {
    output_file=$1
    shift
    {
        echo "captured_at=$(date --utc --iso-8601=ns)"
        printf "command="
        printf "%q " "$@"
        echo
        "$@"
        echo "exit_status=$?"
    } >"$output_dir/$output_file" 2>&1
}

run_raw() {
    output_file=$1
    shift
    "$@" >"$output_dir/$output_file" 2>"$output_dir/$output_file.stderr.log"
    command_status=$?
    {
        echo "captured_at=$(date --utc --iso-8601=ns)"
        printf "command="
        printf "%q " "$@"
        echo
        echo "exit_status=$command_status"
    } >"$output_dir/$output_file.metadata.txt"
}

run_capture date_utc.txt date --utc --iso-8601=ns
run_capture uname.txt uname -a
run_capture os_release.txt cat /etc/os-release
run_capture hostname.txt hostname
run_capture lscpu.txt lscpu
run_raw lscpu.json lscpu --json
run_capture memory.txt free -b
run_capture proc_meminfo.txt cat /proc/meminfo
run_capture mounts.txt findmnt
run_capture workspace_mount.txt findmnt /workspace
run_capture workspace_disk_usage.txt df -B1 /workspace
run_capture block_devices.txt lsblk -O
run_capture pci_devices.txt lspci -nnvv
run_capture loaded_modules.txt lsmod

run_capture nvidia_smi.txt nvidia-smi
run_capture nvidia_smi_query.txt nvidia-smi \
    --query-gpu=timestamp,name,uuid,pci.bus_id,vbios_version,inforom.img,driver_version,pstate,compute_mode,persistence_mode,memory.total,memory.used,memory.free,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics,clocks.max.sm,clocks.max.memory,temperature.gpu,fan.speed,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
    --format=csv
run_capture nvidia_smi_q.txt nvidia-smi -q
run_raw nvidia_smi_q.xml nvidia-smi -q -x
run_capture nvidia_smi_topology.txt nvidia-smi topo -m
run_capture nvidia_smi_supported_clocks.txt nvidia-smi -q -d SUPPORTED_CLOCKS
run_capture nvidia_smi_help_query_gpu.txt nvidia-smi --help-query-gpu
run_capture nvidia_driver_proc.txt cat /proc/driver/nvidia/version
run_capture nvcc_version.txt nvcc --version

run_capture python_version.txt "$python_executable" -VV
run_capture python_sysconfig.txt "$python_executable" -m sysconfig
run_capture pip_version.txt "$python_executable" -m pip --version
run_capture pip_freeze.txt "$python_executable" -m pip freeze --all
run_raw pip_list.json "$python_executable" -m pip list --format=json
run_raw pip_inspect.json "$python_executable" -m pip inspect
run_capture pip_debug.txt "$python_executable" -m pip debug --verbose
run_capture torch_collect_env.txt "$python_executable" -m torch.utils.collect_env
run_capture torch_config.txt "$python_executable" -c \
    'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "arch_list": torch.cuda.get_arch_list(), "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "device_properties": str(torch.cuda.get_device_properties(0)), "torch_config": torch.__config__.show()}, indent=2))'

if [[ -d "$repository_root/.git" ]]; then
    run_capture git_commit.txt git -C "$repository_root" rev-parse HEAD
    run_capture git_status.txt git -C "$repository_root" status --short
    run_raw git_diff.patch git -C "$repository_root" diff --binary
    run_capture git_tracked_files.txt git -C "$repository_root" ls-files -s
fi

find "$repository_root/projects/olm-mfu-benchmark" \
    -type f \
    ! -path '*/__pycache__/*' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$output_dir/benchmark_source_sha256.txt" 2>&1

echo "manifest_completed_at=$(date --utc --iso-8601=ns)" \
    >"$output_dir/manifest_complete.txt"
