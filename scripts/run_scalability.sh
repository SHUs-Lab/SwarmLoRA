#!/usr/bin/env bash
# Adapter pool scalability sweep (10..500 adapters). Usage: bash scripts/run_scalability.sh [pool_size|all]
set -euo pipefail
trap 'kill_all 2>/dev/null; exit 130' INT TERM
trap 'kill_all 2>/dev/null' EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
CLUSTER_DIR="$PROJECT_ROOT/src/controller/cluster"
CLUSTER_SH="$CLUSTER_DIR/cluster.sh"
PYTHON="$PROJECT_ROOT/venv/bin/python"

if [[ ! -f "$CLUSTER_DIR/cluster.conf" ]]; then
    echo "[scalability] ERROR: $CLUSTER_DIR/cluster.conf not found."
    echo "[scalability]        This experiment requires a 2-node cluster (see README's Hardware Requirements)."
    echo "[scalability]        Copy the template and fill in your own node addresses:"
    echo "[scalability]          cp $CLUSTER_DIR/cluster.conf.example $CLUSTER_DIR/cluster.conf"
    echo "[scalability]          \$EDITOR $CLUSTER_DIR/cluster.conf"
    exit 1
fi
source "$CLUSTER_DIR/cluster.conf"

POOL_SIZE="${1:?Usage: bash scripts/run_scalability.sh <pool_size|all>}"
RESULTS_BASE="./benchmark_results/scalability"
TRACES_DIR="traces/Scalability"
PREWARM_WORKERS="${PREWARM_WORKERS:-20}"

# Remote node config (set these env vars for your cluster)
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST env var (e.g. user@1.2.3.4)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_ROOT="${REMOTE_ROOT:?Set REMOTE_ROOT env var (project root on remote node)}"

log() { echo "[scalability] $(date '+%H:%M:%S') $*"; }

kill_all() {
    # Kill controllers/agents first (they respawn workers)
    pkill -9 -f "controller.controller\|node_agent\|global_controller\|controller.benchmark" 2>/dev/null || true
    sleep 1
    # Kill GPU processes (workers, aggregators, template)
    # Reap our GPU processes by BINARY PATH, not by nvidia-smi pid: inside a
    # container nvidia-smi reports HOST-namespace pids that ps cannot resolve,
    # so any owner check silently skips every one and nothing is cleaned up.
    # Forked workers and the template also do not match the name patterns above.
    # Matching on $PYTHON also scopes the kill to this checkout's interpreter,
    # so a shared box's other GPU jobs are left alone.
    # Bracket the first char so this pattern never matches its own cmdline.
    PYTHON_PAT="[${PYTHON:0:1}]${PYTHON:1}"
    pkill -9 -f "$PYTHON_PAT" 2>/dev/null || true
    sleep 2
    pkill -9 nvidia-cuda-mps 2>/dev/null || true
    # Same on remote
    ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o BatchMode=yes $REMOTE_HOST \
        'killall -9 python 2>/dev/null; sleep 1; pkill -9 nvidia-cuda-mps 2>/dev/null' 2>/dev/null || true
    sleep 3
}

sync_pool() {
    local pool=$1
    local adapter_prefix="${ADAPTER_PREFIX:-../sim-adapters/pool-${pool}/lora-}"
    # Derive pool dir from adapter prefix (strip trailing "lora-")
    local src="${adapter_prefix%lora-}"
    local dst="${REMOTE_ROOT}/${src#../}"

    # Check if pool already exists on remote
    local remote_count
    remote_count=$(ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o BatchMode=yes $REMOTE_HOST \
        "ls ${dst} 2>/dev/null | wc -l" 2>/dev/null) || remote_count=0
    if [[ "$remote_count" -gt 0 ]]; then
        log "Pool already on remote (${remote_count} entries), skipping sync"
        return 0
    fi

    log "Syncing ${src} to remote:${dst}..."
    rsync -avzL -e "ssh -p $REMOTE_PORT" "$src" "${REMOTE_HOST}:${dst}" 2>&1 | tail -2
    log "Pool synced ($(du -sh $src | cut -f1))"
}

run_pool() {
    local pool=$1
    local trace_file="${TRACES_DIR}/poisson_rps3_10min_${pool}adp.jsonl"
    local results_dir="${RESULTS_BASE}/pool_${pool}"
    local adapter_prefix="${ADAPTER_PREFIX:-../sim-adapters/pool-${pool}/lora-}"

    if [[ ! -f "$trace_file" ]]; then
        log "ERROR: Trace not found: $trace_file"
        return 1
    fi

    local req_count=$(wc -l < "$trace_file")

    log "============================================================"
    log "SCALABILITY: pool-${pool} (${req_count} requests, Poisson 3 RPS)"
    log "  Trace:    $trace_file"
    log "  Adapters: $adapter_prefix"
    log "  Results:  $results_dir"
    log "============================================================"

    # Sync pool to remote
    sync_pool "$pool"

    # Clean and start
    kill_all
    rm -f logs/*.log logs/*.sock
    ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no -o BatchMode=yes $REMOTE_HOST \
        "rm -f ${REMOTE_ROOT}/multiple_gpus/logs/*.log" 2>/dev/null
    mkdir -p "$results_dir"

    # Start cluster
    ADAPTER_PREFIX="$adapter_prefix" ADAPTERS_PER_MODEL="$pool" bash "$CLUSTER_SH" start

    # Prewarm
    log "Prewarming ${PREWARM_WORKERS} workers..."
    local prewarm_adapter="${adapter_prefix}0"

    # Master prewarm
    local master_share=$((PREWARM_WORKERS / 2))
    local remote_share=$((PREWARM_WORKERS - master_share))

    curl -sf -X POST "http://localhost:9100/prewarm" \
        -H "Content-Type: application/json" \
        -d "{\"adapters\": {\"${prewarm_adapter}\": ${master_share}}}" \
        --max-time 300 2>/dev/null &

    # Remote prewarm via tunnel
    curl -sf -X POST "http://localhost:9101/prewarm" \
        -H "Content-Type: application/json" \
        -d "{\"adapters\": {\"${prewarm_adapter}\": ${remote_share}}}" \
        --max-time 300 2>/dev/null &

    wait
    log "Prewarm complete"

    # Run benchmark
    log "Running benchmark..."
    "$PYTHON" -m controller.benchmark \
        --api-url "http://localhost:${GLOBAL_CONTROLLER_PORT}" \
        --trace-file "$trace_file" \
        --adapter-map-prefix "$adapter_prefix" \
        --output-dir "$results_dir"

    # Copy results
    local latest_summary=$(ls -t "$results_dir"/summary_*.json 2>/dev/null | head -1)
    local latest_records=$(ls -t "$results_dir"/records_*.json 2>/dev/null | head -1)
    [[ -n "$latest_summary" ]] && cp "$latest_summary" "$results_dir/summary.json"
    [[ -n "$latest_records" ]] && cp "$latest_records" "$results_dir/records.json"

    # Print results
    if [[ -f "$results_dir/summary.json" ]]; then
        log "============ RESULTS: pool-${pool} ============"
        "$PYTHON" -c "
import json, numpy as np
d = json.load(open('$results_dir/summary.json'))
recs = json.load(open('$results_dir/records.json'))
ok = [r for r in recs if r['success']]
sys_ttft = [r['ttft_ms'] - r['routing_ms'] for r in ok]
user_ttft = [r['ttft_ms'] for r in ok]
tpot = [1000/r['decode_throughput_tps'] for r in ok if r['decode_throughput_tps']>0]
route = [r['routing_ms'] for r in ok]
e2e = [r['e2e_ms']/1000 for r in ok]
cold = sum(1 for r in ok if r.get('cold_start'))
p = lambda a,x: np.percentile(a,x)
print(f'  Reqs: {d[\"successful_requests\"]}/{d[\"total_requests\"]} fail={d[\"failed_requests\"]} cold={cold}')
print(f'  Sys TTFT:  avg={np.mean(sys_ttft):.0f}ms p50={p(sys_ttft,50):.0f}ms p90={p(sys_ttft,90):.0f}ms')
print(f'  User TTFT: avg={np.mean(user_ttft):.0f}ms p50={p(user_ttft,50):.0f}ms p90={p(user_ttft,90):.0f}ms')
print(f'  TPOT:      avg={np.mean(tpot):.1f}ms p50={p(tpot,50):.1f}ms p90={p(tpot,90):.1f}ms')
print(f'  E2E:       avg={np.mean(e2e):.1f}s p50={p(e2e,50):.1f}s p90={p(e2e,90):.1f}s')
print(f'  Route:     avg={np.mean(route):.0f}ms p50={p(route,50):.0f}ms p90={p(route,90):.0f}ms')
duration = d[\"end_time\"] - d[\"start_time\"]
rps = d[\"successful_requests\"] / duration if duration > 0 else 0
print(f'  Throughput: {d[\"tokens_per_second\"]:.0f} t/s  req/s: {rps:.2f}')
"
    fi

    # Cleanup
    kill_all
    log "Done: pool-${pool}"
}

# Main
if [[ "$POOL_SIZE" == "all" ]]; then
    for pool in 10 20 50 100 200 500; do
        run_pool "$pool"
        echo ""
    done

    log "============================================================"
    log "ALL SCALABILITY RUNS COMPLETE"
    log "============================================================"
else
    run_pool "$POOL_SIZE"
fi
