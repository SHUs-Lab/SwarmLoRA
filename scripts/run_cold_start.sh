#!/usr/bin/env bash
# Measure worker spawn and swap times (Fig 6).
# Usage: bash scripts/run_cold_start.sh [mode] [count]
#   mode:  spawn | swap | both (default: both)
#   count: number of workers to spawn/swap (default: 10)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM

MODE="${1:-both}"
COUNT="${2:-${COUNT:-10}}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmark_results/cold_start}"
mkdir -p "$RESULTS_DIR"

log "============================================================"
log "COLD START BENCHMARK: mode=$MODE, count=$COUNT"
log "============================================================"

stop_services
start_mps

# Start aggregator
log "Starting aggregator on $AGG_DEVICE..."
agg_mps_pipe="/tmp/mps_${AGG_DEVICE#cuda:}"
nohup env CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY="$agg_mps_pipe" \
    "$PYTHON" src/aggregator.py \
    --device "cuda:0" --port "$AGG_TCP_PORT" --health-port "$AGG_HEALTH_PORT" \
    > "$LOG_DIR/aggregator_0.log" 2>&1 &

wait_for_health "http://localhost:${AGG_HEALTH_PORT}/health" "Aggregator" "$AGG_HEALTH_TIMEOUT"

# Start controller
log "Starting controller (scheduler=$SCHEDULER, scale_down=600s)..."
nohup "$PYTHON" -m controller.controller \
    --port "$CTRL_PORT" \
    --aggregator-port-map "$AGG_PORT_MAP" \
    --aggregator-health-port-map "$AGG_HEALTH_PORT_MAP" \
    --scheduler "$SCHEDULER" \
    --scale-down-delay 600 \
    > "$LOG_DIR/controller.log" 2>&1 &

wait_for_health "http://localhost:${CTRL_PORT}/health" "Controller" "$CTRL_HEALTH_TIMEOUT"

# Prewarm workers for swap-only mode
if [[ "$MODE" == "swap" ]]; then
    log "Prewarming $COUNT workers for swap benchmark..."
    ADAPTER="${ADAPTER_PREFIX}0"
    curl -sf -X POST "http://localhost:${CTRL_PORT}/prewarm" \
        -H "Content-Type: application/json" \
        -d "{\"adapters\": {\"${ADAPTER}\": ${COUNT}}}" \
        --max-time 300 || log "WARN: Prewarm failed"
fi

# Run benchmark
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_FILE="$RESULTS_DIR/cold_start_${MODE}_${COUNT}_${TIMESTAMP}.json"

log "Running cold start benchmark..."
"$PYTHON" benchmarks/cold_start.py \
    --mode "$MODE" \
    --count "$COUNT" \
    --api-url "http://localhost:${CTRL_PORT}" \
    --adapter-prefix "$ADAPTER_PREFIX" \
    --output "$OUTPUT_FILE"

log "Results saved to $OUTPUT_FILE"

# Cleanup
stop_services
bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
log "Done."
