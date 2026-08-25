#!/usr/bin/env bash
# run_cold_start.sh — Measure single-worker cold start (Fig. 8's ServerlessLoRA bar).
#
# Starts BaseModelServers (one per GPU), spawns ONE worker, and measures the
# time from Popen to /health returning "ready".
#
# Usage:
#   bash scripts/run_cold_start.sh [config]
#   bash scripts/run_cold_start.sh deployment_config_pckp_30.yaml

set -euo pipefail

cd "$(dirname "$0")/.."
source sless-venv/bin/activate

CONFIG_TEMPLATE="${1:-${CONFIG:-deployment_config_pckp_30.yaml}}"
RUN_INFERENCE="${2:-${RUN_INFERENCE:-}}"  # pass "inference" to also measure first inference
CONFIG="$(mktemp /tmp/serverlesslora_config_XXXXXX.yaml)"
python3 scripts/adapt_gpu_config.py "$CONFIG_TEMPLATE" "$CONFIG"

RESULTS_DIR="${RESULTS_DIR:-benchmark_results/cold_start}"
mkdir -p "$RESULTS_DIR"
mkdir -p logs

# Start per-GPU MPS daemons (ignore if already running)
./scripts/setup_mps.sh start || true

cleanup() {
    echo "[$(date +%H:%M:%S)] Cleaning up..."
    pkill -9 -f "worker_batched|base_model_server" 2>/dev/null || true
    sleep 2
}

cleanup_mps() {
    echo "[$(date +%H:%M:%S)] Stopping MPS daemons..."
    ./scripts/setup_mps.sh stop
}

cleanup_config() {
    rm -f "$CONFIG"
}

trap 'cleanup; cleanup_mps; cleanup_config' EXIT
cleanup

echo "============================================================"
echo " Cold Start Benchmark (single worker) — $(date)"
echo " Config:     $CONFIG"
echo "============================================================"
echo ""

# Clear old logs
find logs/ -name 'cold_bench_*.log' -delete 2>/dev/null || true
find logs/ -name 'bms_*.log' -delete 2>/dev/null || true

# Parse GPU config and start BaseModelServers
echo "[$(date +%H:%M:%S)] Starting BaseModelServers..."
BMS_PIDS=()

# Extract GPU info from config
GPU_INFO=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for gpu in cfg.get('gpus', []):
    print(f\"{gpu['device_id']} {gpu['base_model_server_port']}\")
")

while read -r DEVICE_ID BMS_PORT; do
    echo "  Starting BMS on GPU $DEVICE_ID, port $BMS_PORT..."
    python base_model_server.py \
        --device "cuda:${DEVICE_ID}" \
        --port "$BMS_PORT" \
        > "logs/bms_cold_gpu${DEVICE_ID}.log" 2>&1 &
    BMS_PIDS+=($!)
done <<< "$GPU_INFO"

# Wait for BMS to be ready (BMS uses raw TCP, not HTTP)
echo "[$(date +%H:%M:%S)] Waiting for BaseModelServers..."
while read -r DEVICE_ID BMS_PORT; do
    for i in $(seq 1 60); do
        if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('localhost', int(sys.argv[1])))
    s.sendall(b'STATUS')
    s.recv(4096)
    s.close()
except:
    s.close()
    sys.exit(1)
" "$BMS_PORT" 2>/dev/null; then
            echo "  BMS GPU $DEVICE_ID ready (port $BMS_PORT)."
            break
        fi
        sleep 2
    done
done <<< "$GPU_INFO"
echo ""

# Build inference flag
INFERENCE_FLAG=""
if [ "${RUN_INFERENCE}" = "inference" ]; then
    INFERENCE_FLAG="--run-inference"
fi

# Single-worker cold start (Fig. 8's ServerlessLoRA number)
echo "[$(date +%H:%M:%S)] Running single worker cold start..."
python tools/cold_start_benchmark.py \
    --config "$CONFIG" \
    --num-workers 1 \
    --concurrent 1 \
    $INFERENCE_FLAG \
    --output "${RESULTS_DIR}/cold_start_1w_1c.json" \
    2>&1

echo ""
echo "[$(date +%H:%M:%S)] Done."
