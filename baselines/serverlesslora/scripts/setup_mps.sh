#!/usr/bin/env bash
# setup_mps.sh — Per-GPU MPS daemon management.
#
# Launches one nvidia-cuda-mps-control daemon per GPU, each with its own
# pipe directory (/tmp/mps_{gpu_idx}).  Client processes select a physical
# GPU by setting CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_{N} and
# CUDA_VISIBLE_DEVICES=0 (the MPS server remaps the device).
#
# Usage:
#   ./setup_mps.sh start   [gpu_count]   # default: auto-detect
#   ./setup_mps.sh stop    [gpu_count]
#   ./setup_mps.sh status  [gpu_count]

set -euo pipefail

ACTION="${1:-status}"
GPU_COUNT="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

case "$ACTION" in
    start)
        # Kill any existing system-wide MPS daemon first
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        sleep 1

        for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
            pipe_dir="/tmp/mps_${gpu_idx}"
            log_dir="/tmp/mps_log_${gpu_idx}"
            mkdir -p "$pipe_dir" "$log_dir"

            if [ -S "$pipe_dir/control" ] 2>/dev/null && \
               echo "get_server_list" | CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" nvidia-cuda-mps-control >/dev/null 2>&1; then
                echo "  [--] GPU $gpu_idx: MPS daemon already running, reusing (pipe: $pipe_dir)"
                continue
            fi
            # A socket file can outlive its daemon (stop doesn't always unlink
            # the pipe), so remove it and let the daemon bind a fresh one.
            rm -f "$pipe_dir/control"

            CUDA_VISIBLE_DEVICES=$gpu_idx \
            CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" \
            CUDA_MPS_LOG_DIRECTORY="$log_dir" \
                nvidia-cuda-mps-control -d 2>/dev/null || true

            echo "  [OK] MPS daemon started for GPU $gpu_idx (pipe: $pipe_dir)"
        done
        echo "Started $GPU_COUNT per-GPU MPS daemons."
        ;;

    stop)
        for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
            pipe_dir="/tmp/mps_${gpu_idx}"
            if [ -S "$pipe_dir/control" ] 2>/dev/null; then
                echo quit | CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" nvidia-cuda-mps-control 2>/dev/null || true
                echo "  [OK] MPS daemon stopped for GPU $gpu_idx"
            else
                echo "  [--] GPU $gpu_idx: no MPS daemon running"
            fi
        done
        # Also kill system-wide daemon if present
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        echo "Stopped MPS daemons."
        ;;

    status)
        for gpu_idx in $(seq 0 $((GPU_COUNT - 1))); do
            pipe_dir="/tmp/mps_${gpu_idx}"
            if [ -S "$pipe_dir/control" ] 2>/dev/null; then
                active=$(CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" nvidia-cuda-mps-control -c "get_default_active_thread_percentage" 2>/dev/null || echo "?")
                echo "  GPU $gpu_idx: RUNNING (pipe: $pipe_dir, active_threads: $active)"
            else
                echo "  GPU $gpu_idx: NOT RUNNING"
            fi
        done
        ;;

    *)
        echo "Usage: $0 {start|stop|status} [gpu_count]"
        exit 1
        ;;
esac
