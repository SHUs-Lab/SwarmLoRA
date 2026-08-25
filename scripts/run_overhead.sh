#!/usr/bin/env bash
# IPC overhead: baseline TPOT then split-architecture profiling at N=1..32 (Fig 12).
# Usage: bash scripts/run_overhead.sh [worker_counts]
#   worker_counts: comma-separated, default: 1,4,8,12,16,20,24,28,32
#   (32 is the L40S memory ceiling; raise it on a larger GPU.)
#   env overrides: DECODE_TOKENS (default 100), REPEATS (default 3),
#                  ADAPTER (default ../sim-adapters/pool-10-r16/lora-0), GPU (default 0)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM
trap stop_services EXIT

WORKERS="${1:-1,4,8,12,16,20,24,28,32}"
DECODE_TOKENS="${DECODE_TOKENS:-100}"
REPEATS="${REPEATS:-3}"
ADAPTER="${ADAPTER:-../sim-adapters/pool-10-r16/lora-0}"
GPU="${GPU:-0}"

# Sweep exceeds MAX_WORKERS_PER_GPU=22, so size the slot pool to the largest N.
NUM_SLOTS=$(echo "$WORKERS" | tr ',' '\n' | sort -n | tail -1)

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmark_results}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "$RESULTS_DIR"

AGG_TCP_PORT=$("$PYTHON" -c "import config; print(config.AGGREGATOR_PORTS.get($GPU, 50056+$GPU))")
AGG_HEALTH_PORT=$("$PYTHON" -c "import config; print(config.AGGREGATOR_HEALTH_PORTS.get($GPU, 8000+$GPU))")
MPS_PIPE="/tmp/mps_${GPU}"

# ── Main ──────────────────────────────────────────────────────────────────────
log "============================================================"
log "RQ5: IPC OVERHEAD OF SPLIT ARCHITECTURE"
log "  Workers: $WORKERS"
log "  GPU: cuda:$GPU"
log "  Decode tokens: $DECODE_TOKENS"
log "  Repeats: $REPEATS"
log "  Adapter: $ADAPTER"
log "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE — Standard LoRA inference
# ══════════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════"
log "PHASE 1: Baseline (standard LoRA, no split)"
log "════════════════════════════════════════════════════"

stop_services
# No MPS needed for baseline — just one process
bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
sleep 1

BASELINE_FILE="$RESULTS_DIR/baseline_tpot_${TIMESTAMP}.json"

env CUDA_VISIBLE_DEVICES=$GPU \
    "$PYTHON" benchmarks/baseline_tpot.py \
    --adapter "$ADAPTER" \
    --device "cuda:0" \
    --decode-tokens "$DECODE_TOKENS" \
    --repeats "$REPEATS" \
    --output "$BASELINE_FILE"

log "Baseline saved to $BASELINE_FILE"

# Free GPU memory from baseline
kill_component "Baseline" "baseline_tpot" 2>/dev/null || true
# Force-kill any remaining python on GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 3

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SPLIT — Our architecture with profiling
# ══════════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════"
log "PHASE 2: Split architecture (aggregator + workers)"
log "════════════════════════════════════════════════════"

# Setup MPS
bash "$SCRIPT_DIR/setup_mps.sh" start

# Start Aggregator
log "Starting aggregator on cuda:$GPU..."
log "Slot pool sized to $NUM_SLOTS (max of tested worker counts)"
nohup env CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    "$PYTHON" src/aggregator.py \
    --device "cuda:0" --port "$AGG_TCP_PORT" --health-port "$AGG_HEALTH_PORT" \
    --num-slots "$NUM_SLOTS" \
    > "$LOG_DIR/aggregator_${GPU}.log" 2>&1 &
AGG_PID=$!
log "Aggregator PID: $AGG_PID"

wait_for_health "http://localhost:${AGG_HEALTH_PORT}/health" "Aggregator" 180

SPLIT_FILE="$RESULTS_DIR/ipc_overhead_${TIMESTAMP}.json"

log "Running split architecture benchmark..."
env CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    PROFILE_LAYERS=1 \
    "$PYTHON" benchmarks/ipc_overhead.py \
    --workers "$WORKERS" \
    --decode-tokens "$DECODE_TOKENS" \
    --adapter "$ADAPTER" \
    --agg-host localhost \
    --agg-port "$AGG_TCP_PORT" \
    --agg-health-port "$AGG_HEALTH_PORT" \
    --device "cuda:0" \
    --repeats "$REPEATS" \
    --output "$SPLIT_FILE"

log "Split results saved to $SPLIT_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════"
log "PHASE 3: Overhead comparison"
log "════════════════════════════════════════════════════"

"$PYTHON" -c "
import json, numpy as np

baseline = json.load(open('$BASELINE_FILE'))
split = json.load(open('$SPLIT_FILE'))

baseline_tpot = baseline['tpot_avg_ms']

print()
print('=' * 72)
print('RQ5: IPC Overhead of Split Architecture')
print('=' * 72)
print()
print(f'Baseline TPOT (standard LoRA): {baseline_tpot:.2f} ms')
print()

cols = sorted(split['results'].keys(), key=lambda x: int(x))
col_w = 12

header = f\"{'':>12}\" + ''.join(f\"{'N='+c:>{col_w}}\" for c in cols)
print(header)
print('-' * len(header))

row = f\"{'GATHER':>12}\"
for c in cols:
    row += f\"{split['results'][c]['gather_ms']:>{col_w-2}.2f}ms\"
print(row)

row = f\"{'SCATTER':>12}\"
for c in cols:
    row += f\"{split['results'][c]['scatter_ms']:>{col_w-2}.2f}ms\"
print(row)

row = f\"{'COMPUTE':>12}\"
for c in cols:
    d = split['results'][c]
    comp = d['total_ms'] - d['gather_ms'] - d['scatter_ms']
    row += f\"{comp:>{col_w-2}.2f}ms\"
print(row)

print('-' * len(header))

row = f\"{'TOTAL':>12}\"
for c in cols:
    row += f\"{split['results'][c]['total_ms']:>{col_w-2}.2f}ms\"
print(row)

row = f\"{'IPC transfer':>12}\"
for c in cols:
    d = split['results'][c]
    ipc = d['gather_ms'] + d['scatter_ms']
    row += f\"{ipc:>{col_w-2}.2f}ms\"
print(row)

row = f\"{'IPC %':>12}\"
for c in cols:
    row += f\"{split['results'][c]['ipc_pct']:>{col_w-2}.1f}% \"
print(row)

print()
print('Overhead breakdown (N=1 vs baseline):')
n1 = split['results']['1']
total_overhead = n1['total_ms'] - baseline_tpot
ipc_transfer = n1['gather_ms'] + n1['scatter_ms']
sync_overhead = total_overhead - ipc_transfer
print(f'  Baseline TPOT:       {baseline_tpot:>8.2f} ms')
print(f'  Split TPOT (N=1):    {n1[\"total_ms\"]:>8.2f} ms')
print(f'  ─────────────────────────────')
print(f'  Total overhead:      {total_overhead:>8.2f} ms ({total_overhead/baseline_tpot*100:.1f}%)')
print(f'    IPC transfer:      {ipc_transfer:>8.2f} ms  (GATHER + SCATTER)')
print(f'    Sync overhead:     {sync_overhead:>8.2f} ms  (barriers + pipeline stalls)')
print()
print('GATHER  = Worker → IPC buffer P2P copy (130 barriers/token)')
print('SCATTER = IPC buffer → Worker P2P copy (130 barriers/token)')
print('COMPUTE = LoRA + base GEMM + attention + norms (overlapping)')
"

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
log "Cleaning up..."
stop_services
bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
log "Done."
