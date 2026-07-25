#!/usr/bin/env bash
# Throughput scaling: 1..N workers on a single GPU (Fig 2, motivation).
# Usage: bash scripts/run_scaling.sh [worker_counts]
#   worker_counts: comma-separated, default: 1,2,4,8,12,16,24
#   env overrides: MAX_TOKENS (default 128), REPEATS (default 3),
#                  ADAPTER (default ../sim-adapters/pool-10-r16/lora-0), GPU (default 0)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM

WORKERS="${1:-1,2,4,8,12,16,24}"
MAX_TOKENS="${MAX_TOKENS:-128}"
REPEATS="${REPEATS:-3}"
ADAPTER="${ADAPTER:-../sim-adapters/pool-10-r16/lora-0}"
GPU="${GPU:-0}"

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmark_results/throughput_scaling}"
mkdir -p "$RESULTS_DIR"

AGG_TCP_PORT=$("$PYTHON" -c "import config; print(config.AGGREGATOR_PORTS.get($GPU, 50056+$GPU))")
AGG_HEALTH_PORT=$("$PYTHON" -c "import config; print(config.AGGREGATOR_HEALTH_PORTS.get($GPU, 8000+$GPU))")
MPS_PIPE="/tmp/mps_${GPU}"

# ── Main ──────────────────────────────────────────────────────────────────────
log "============================================================"
log "THROUGHPUT SCALING BENCHMARK"
log "  Workers: $WORKERS"
log "  GPU: cuda:$GPU"
log "  Max tokens: $MAX_TOKENS"
log "  Repeats: $REPEATS (best of N)"
log "============================================================"

# ── Cleanup ───────────────────────────────────────────────────────────────────
log "Cleaning up stale processes..."
stop_services

# ── Setup MPS ─────────────────────────────────────────────────────────────────
bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
sleep 1
bash "$SCRIPT_DIR/setup_mps.sh" start

# ── Start Aggregator ──────────────────────────────────────────────────────────
log "Starting aggregator on cuda:$GPU..."
nohup env CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    "$PYTHON" src/aggregator.py \
    --device "cuda:0" --port "$AGG_TCP_PORT" --health-port "$AGG_HEALTH_PORT" \
    > "$LOG_DIR/aggregator_${GPU}.log" 2>&1 &
AGG_PID=$!
log "Aggregator PID: $AGG_PID"

wait_for_health "http://localhost:${AGG_HEALTH_PORT}/health" "Aggregator" 180

# ── Run Benchmark ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_FILE="$RESULTS_DIR/scaling_${TIMESTAMP}.json"

log "Running throughput scaling benchmark..."
env CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    "$PYTHON" benchmarks/throughput_scaling.py \
    --workers "$WORKERS" \
    --max-tokens "$MAX_TOKENS" \
    --adapter "$ADAPTER" \
    --agg-host localhost \
    --agg-port "$AGG_TCP_PORT" \
    --agg-health-port "$AGG_HEALTH_PORT" \
    --device "cuda:0" \
    --repeats "$REPEATS" \
    --output "$OUTPUT_FILE"

log "Results saved to $OUTPUT_FILE"

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
log "Cleaning up..."
stop_services
bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
log "Done."
