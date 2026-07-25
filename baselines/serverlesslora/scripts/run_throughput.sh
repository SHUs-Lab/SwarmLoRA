#!/usr/bin/env bash
# run_throughput.sh — PW-20, 22 workers/GPU, Poisson sustained throughput test (Fig. 5)
#
# Usage:
#   bash scripts/run_throughput.sh            # run all 5 rates (matches SwarmLoRA's RQ1)
#   bash scripts/run_throughput.sh rps4 rps8  # run specific rates

set -euo pipefail

cd "$(dirname "$0")/.."
source sless-venv/bin/activate

# Start per-GPU MPS daemons
./scripts/setup_mps.sh start

CONFIG_TEMPLATE="${CONFIG:-deployment_config_pckp_25.yaml}"
CONFIG="$(mktemp /tmp/serverlesslora_config_XXXXXX.yaml)"
python3 scripts/adapt_gpu_config.py "$CONFIG_TEMPLATE" "$CONFIG"
CONTROLLER="${CONTROLLER:-http://localhost:8000}"
SLO_MS="${SLO_MS:-2000}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
GPU_PRICE="${GPU_PRICE:-1.50}"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results/throughput}"
TRACES_DIR="${TRACES_DIR:-../../traces/3min_poisson}"

ALL_RATES=(rps1 rps2 rps4 rps8 rps12)

if [ $# -gt 0 ]; then
    RATES=("$@")
else
    RATES=("${ALL_RATES[@]}")
fi

mkdir -p "$RESULTS_DIR"
mkdir -p logs

cleanup() {
    echo "[$(date +%H:%M:%S)] Shutting down cluster..."
    python launch_deployment.py --config "$CONFIG" --shutdown 2>/dev/null || true
    sleep 2
    pkill -9 -f "worker_batched|controller.py|preload_agent|preload_scheduler|base_model_server" 2>/dev/null || true
    sleep 3
}

cleanup_mps() {
    echo "[$(date +%H:%M:%S)] Stopping per-GPU MPS daemons..."
    ./scripts/setup_mps.sh stop
}

cleanup_config() {
    rm -f "$CONFIG"
}

wait_for_containers() {
    local expected
    expected=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cpf = cfg.get('defaults', {}).get('containers_per_function', 1)
total = sum(fn.get('containers', cpf) for fn in cfg.get('functions', []))
print(total)
")
    local max_wait=300
    local elapsed=0
    local ready=0
    echo "  Expecting $expected containers to be ready..."
    while [ $elapsed -lt $max_wait ]; do
        ready=$(curl -s --max-time 5 "$CONTROLLER/containers" 2>/dev/null \
            | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(sum(1 for c in d.get('containers', []) if c.get('status') == 'ready'))
except:
    print(0)
" 2>/dev/null)
        ready=${ready:-0}
        if [ "$ready" -ge "$expected" ]; then
            echo "  All $ready containers ready."
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  Containers ready: $ready/$expected (${elapsed}s)"
    done
    echo "WARNING: Only $ready/$expected containers ready after ${max_wait}s, proceeding anyway"
    return 0
}

trap 'cleanup; cleanup_mps; cleanup_config' EXIT
cleanup  # defensive: clear out anything left over from a previous run before we start

echo "============================================================"
echo " ServerlessLoRA Poisson Throughput Test (3min) — $(date)"
echo " Config: $CONFIG"
echo " Workers/GPU: 22, Batch window: 250ms, Max batch: 16"
echo " Queue: LIFO, Timeout: 8s"
echo " Rates: ${RATES[*]}"
echo "============================================================"
echo ""

for rate in "${RATES[@]}"; do
    trace_file="${TRACES_DIR}/poisson_${rate}_3min_10adp.jsonl"
    if [ ! -f "$trace_file" ]; then
        echo "SKIP: $trace_file not found"
        continue
    fi

    echo "============================================================"
    echo " RATE: $rate"
    echo "============================================================"

    # Fresh cluster between rates
    [[ "$rate" != "${RATES[0]}" ]] && cleanup

    echo "[$(date +%H:%M:%S)] Launching fresh cluster..."
    find logs/ -name '*.log' -delete 2>/dev/null
    LAUNCH_ARGS=(--config "$CONFIG" --no-wait --skip-profiling)
    python launch_deployment.py "${LAUNCH_ARGS[@]}" 2>&1 | tee "logs/launch.log"
    echo ""

    echo "[$(date +%H:%M:%S)] Waiting for all containers to be ready..."
    wait_for_containers
    echo ""

    echo "[$(date +%H:%M:%S)] Running trace: $rate"
    python tools/trace_replayer.py \
        --trace "$trace_file" \
        --controller "$CONTROLLER" \
        --max-concurrent 512 \
        --output "${RESULTS_DIR}/${rate}_run.json" \
        2>&1
    echo ""

    echo "[$(date +%H:%M:%S)] Collecting metrics..."
    python tools/metrics_collector.py \
        --input "${RESULTS_DIR}/${rate}_run.json" \
        --slo-ms "$SLO_MS" \
        --num-gpus "$NUM_GPUS" \
        --gpu-price-per-hour "$GPU_PRICE" \
        --output "${RESULTS_DIR}/${rate}_metrics.json" \
        --text-report "${RESULTS_DIR}/${rate}_report.txt" \
        --fields throughput \
        2>&1
    echo ""

    echo "[$(date +%H:%M:%S)] Done with $rate"
    echo ""
done

echo "============================================================"
echo " ALL DONE — $(date)"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
echo ""

echo "=== SUMMARY (RQ1: throughput) ==="
printf "%-8s  %-9s\n" "rate" "tok/s"
for rate in "${RATES[@]}"; do
    metrics="${RESULTS_DIR}/${rate}_metrics.json"
    if [ -f "$metrics" ]; then
        python3 -c "
import json
d = json.load(open('$metrics'))
print('%-8s  %-9.1f' % ('$rate', d['tokens_per_second']))
"
    else
        printf "%-8s  NO RESULTS\n" "$rate"
    fi
done
