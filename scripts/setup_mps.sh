#!/usr/bin/env bash
# Per-GPU MPS daemon management. Usage: {start|stop|status}
set -euo pipefail

log() { echo "[MPS] $*"; }

get_gpu_count() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l
}

do_start() {
    local gpu_count
    gpu_count=$(get_gpu_count)
    if [[ "$gpu_count" -eq 0 ]]; then
        log "ERROR: No GPUs detected"
        exit 1
    fi

    local existing_mps
    existing_mps=$(pgrep -f "nvidia-cuda-mps-control" 2>/dev/null || true)
    if [[ -n "$existing_mps" ]]; then
        log "Stopping existing MPS daemon(s)..."
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        for idx in $(seq 0 $((gpu_count - 1))); do
            echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${idx} nvidia-cuda-mps-control 2>/dev/null || true
        done
        sleep 1
        pkill -9 -f "nvidia-cuda-mps-control" 2>/dev/null || true
        pkill -9 -f "nvidia-cuda-mps-server" 2>/dev/null || true
        sleep 1
    fi

    log "Starting per-GPU MPS daemons for $gpu_count GPUs..."

    for gpu_idx in $(seq 0 $((gpu_count - 1))); do
        local pipe_dir="/tmp/mps_${gpu_idx}"
        local log_dir="/tmp/mps_log_${gpu_idx}"
        mkdir -p "$pipe_dir" "$log_dir"

        if [[ -e "${pipe_dir}/control" ]]; then
            log "GPU $gpu_idx: MPS already running (pipe exists), skipping"
            continue
        fi

        CUDA_VISIBLE_DEVICES=$gpu_idx \
        CUDA_MPS_PIPE_DIRECTORY=$pipe_dir \
        CUDA_MPS_LOG_DIRECTORY=$log_dir \
            nvidia-cuda-mps-control -d

        log "GPU $gpu_idx: MPS daemon started (pipe=$pipe_dir)"
    done

    sleep 1
    do_status
}

do_stop() {
    local gpu_count
    gpu_count=$(get_gpu_count)

    log "Stopping per-GPU MPS daemons..."

    for gpu_idx in $(seq 0 $((gpu_count - 1))); do
        local pipe_dir="/tmp/mps_${gpu_idx}"
        if [[ -e "${pipe_dir}/control" ]]; then
            echo quit | CUDA_MPS_PIPE_DIRECTORY=$pipe_dir nvidia-cuda-mps-control 2>/dev/null || true
            log "GPU $gpu_idx: MPS daemon stopped"
        fi
    done

    sleep 1

    pkill -9 -f "nvidia-cuda-mps-control" 2>/dev/null || true
    pkill -9 -f "nvidia-cuda-mps-server" 2>/dev/null || true

    for gpu_idx in $(seq 0 $((gpu_count - 1))); do
        rm -rf "/tmp/mps_${gpu_idx}" "/tmp/mps_log_${gpu_idx}" 2>/dev/null || true
    done

    log "Done."
}

do_status() {
    local gpu_count
    gpu_count=$(get_gpu_count)

    log "=== Per-GPU MPS Status ==="
    for gpu_idx in $(seq 0 $((gpu_count - 1))); do
        local pipe_dir="/tmp/mps_${gpu_idx}"
        if [[ -e "${pipe_dir}/control" ]]; then
            echo "  GPU $gpu_idx: RUNNING (pipe=$pipe_dir)"
        else
            echo "  GPU $gpu_idx: NOT RUNNING"
        fi
    done

    local mps_procs
    mps_procs=$(pgrep -a "nvidia-cuda-mps" 2>/dev/null || true)
    if [[ -n "$mps_procs" ]]; then
        echo ""
        echo "  MPS processes:"
        echo "$mps_procs" | while read -r line; do
            echo "    $line"
        done
    fi
}

case "${1:-help}" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    *)
        echo "Usage: bash scripts/setup_mps.sh {start|stop|status}"
        echo ""
        echo "Starts one MPS daemon per GPU with separate pipe directories."
        echo "Each GPU gets its own 48-client MPS pool."
        echo "Pipe directories: /tmp/mps_{N}/"
        echo ""
        echo "Client usage:"
        echo "  CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_{N} your_command"
        ;;
esac
