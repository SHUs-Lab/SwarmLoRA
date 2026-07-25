#!/usr/bin/env bash
# run_trace_driven.sh — ServerlessLoRA trace-driven benchmark, 10-minute traces
#                        (Figs. 6-7, 9-11, Table II). Config knobs printed at runtime.
#
# Usage:
#   bash scripts/run_trace_driven.sh                    # run all 5 traces
#   bash scripts/run_trace_driven.sh normal bursty_heavy # run specific traces

set -euo pipefail

cd "$(dirname "$0")/.."
source sless-venv/bin/activate

# Start per-GPU MPS daemons (one per GPU instead of system-wide)
./scripts/setup_mps.sh start

CONFIG_TEMPLATE="${CONFIG:-deployment_config_pckp_25.yaml}"
CONFIG="$(mktemp /tmp/serverlesslora_config_XXXXXX.yaml)"
python3 scripts/adapt_gpu_config.py "$CONFIG_TEMPLATE" "$CONFIG"
CONTROLLER="${CONTROLLER:-http://localhost:8000}"
SLO_MS="${SLO_MS:-2000}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
GPU_PRICE="${GPU_PRICE:-1.50}"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results/trace_driven}"
TRACES_DIR="${TRACES_DIR:-../../traces/10min_10adp}"

ALL_TRACES=(steady_light bursty_light normal steady_heavy bursty_heavy)

# Use args as trace names, or run all
if [ $# -gt 0 ]; then
    TRACES=("$@")
else
    TRACES=("${ALL_TRACES[@]}")
fi

mkdir -p "$RESULTS_DIR"
mkdir -p logs

cleanup() {
    echo "[$(date +%H:%M:%S)] Shutting down cluster..."
    python launch_deployment.py --config "$CONFIG" --shutdown 2>/dev/null || true
    sleep 2
    # Force kill any remaining processes
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

# Make sure we start clean
trap 'cleanup; cleanup_mps; cleanup_config' EXIT
cleanup  # defensive: clear out anything left over from a previous run before we start

echo "============================================================"
echo " ServerlessLoRA PW-20/22w Benchmark (10min) — $(date)"
echo " Config: $CONFIG"
echo " Workers/GPU: 22, Batch window: 250ms, Max batch: 16"
echo " Queue: LIFO, Timeout: 8s"
echo " Traces: ${TRACES[*]}"
echo "============================================================"
echo ""

for trace in "${TRACES[@]}"; do
    trace_file="${TRACES_DIR}/${trace}_10min_10adp.jsonl"
    if [ ! -f "$trace_file" ]; then
        echo "SKIP: $trace_file not found"
        continue
    fi

    echo "============================================================"
    echo " TRACE: $trace"
    echo "============================================================"

    # Fresh cluster between traces
    [[ "$trace" != "${TRACES[0]}" ]] && cleanup

    echo "[$(date +%H:%M:%S)] Launching fresh cluster..."
    find logs/ -name '*.log' -delete 2>/dev/null
    LAUNCH_ARGS=(--config "$CONFIG" --no-wait --skip-profiling)
    python launch_deployment.py "${LAUNCH_ARGS[@]}" 2>&1 | tee "logs/launch.log"
    echo ""

    echo "[$(date +%H:%M:%S)] Waiting for all containers to be ready..."
    wait_for_containers
    echo ""

    # Run trace
    echo "[$(date +%H:%M:%S)] Running trace: $trace"
    python tools/trace_replayer.py \
        --trace "$trace_file" \
        --controller "$CONTROLLER" \
        --max-concurrent 512 \
        --output "${RESULTS_DIR}/${trace}_run.json" \
        2>&1
    echo ""

    # Metrics
    echo "[$(date +%H:%M:%S)] Collecting metrics..."
    python tools/metrics_collector.py \
        --input "${RESULTS_DIR}/${trace}_run.json" \
        --slo-ms "$SLO_MS" \
        --num-gpus "$NUM_GPUS" \
        --gpu-price-per-hour "$GPU_PRICE" \
        --output "${RESULTS_DIR}/${trace}_metrics.json" \
        --text-report "${RESULTS_DIR}/${trace}_report.txt" \
        2>&1
    echo ""

    echo "[$(date +%H:%M:%S)] Done with $trace"
    echo ""
done

echo "============================================================"
echo " ALL DONE — $(date)"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
echo ""

# Summary table (RQ3: acceptance, throughput, TTFT/TPOT p50/p90 in ms)
echo "=== SUMMARY (RQ3: trace-driven) ==="
printf "%-14s  %-8s  %-8s  %-9s  %-9s  %-9s  %-9s\n" \
    "trace" "accept%" "tok/s" "ttftP50" "ttftP90" "tpotP50" "tpotP90"
for trace in "${TRACES[@]}"; do
    metrics="${RESULTS_DIR}/${trace}_metrics.json"
    if [ -f "$metrics" ]; then
        python3 -c "
import json
d = json.load(open('$metrics'))
print('%-14s  %-8.1f  %-8.1f  %-9.0f  %-9.0f  %-9.1f  %-9.1f' % (
    '$trace', d['acceptance_rate']*100, d['tokens_per_second'],
    d['ttft_p50_ms'], d['ttft_p90_ms'], d['tpot_p50_ms'], d['tpot_p90_ms']))
"
    else
        printf "%-14s  NO RESULTS\n" "$trace"
    fi
done
