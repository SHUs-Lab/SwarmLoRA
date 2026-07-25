#!/bin/bash
# Cold-start latency reconstruction (Fig. 8 comparison point).
# Best-effort reconstruction -- see cold_start.py's header for provenance.
#
# Usage:
#   One-time setup (converts base model + adapter to sllm_store format):
#     bash baselines/serverlessllm/scripts/run_cold_start.sh --prepare
#   Then measure:
#     bash baselines/serverlessllm/scripts/run_cold_start.sh
#   env overrides: ADAPTER_PATH, STORAGE_PATH (default ./models),
#                  MEM_POOL_SIZE (default 20GB), STORE_PORT (default 8073)
set -euo pipefail
cd "$(dirname "$0")/.."
source sllm-env/bin/activate

# Quiet sllm's own DEBUG/INFO chatter for a cleaner [cold_start] summary.
export LOG_LEVEL="${LOG_LEVEL:-ERROR}"

ADAPTER_PATH="${ADAPTER_PATH:-../../../sim-adapters/pool-10-r16/lora-0}"
STORAGE_PATH="${STORAGE_PATH:-./models}"
OUTPUT="${OUTPUT:-benchmark_results/cold_start/sllm_lora_cold_start_1gpu_reconstructed.json}"
MEM_POOL_SIZE="${MEM_POOL_SIZE:-20GB}"
STORE_PORT="${STORE_PORT:-8073}"

if [[ "${1:-}" == "--prepare" ]]; then
    python3 benchmarks/cold_start.py --prepare \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --adapter-path "$ADAPTER_PATH" \
        --storage-path "$STORAGE_PATH"
else
    # load_model()/load_lora() talk to the sllm-store gRPC server over
    # localhost -- it must be running first (not started implicitly).
    mkdir -p logs
    sllm-store start --storage-path "$STORAGE_PATH" --mem-pool-size "$MEM_POOL_SIZE" \
        > logs/sllm_store_server.log 2>&1 &
    STORE_PID=$!
    trap 'kill "$STORE_PID" 2>/dev/null || true' EXIT

    for i in $(seq 1 30); do
        if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(('localhost', $STORE_PORT))
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    python3 benchmarks/cold_start.py \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --storage-path "$STORAGE_PATH" \
        --num-runs 4 \
        --output "$OUTPUT"
fi
