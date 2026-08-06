# Shared setup and helpers for benchmark scripts. Source this, don't run it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/venv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Host CUDA (nvcc) can be a different major version than the pinned torch wheel's
# bundled runtime, so the compiled extensions may need the host's libcudart.so on
# the loader path. Auto-detected here so it works on any host CUDA version.
if command -v nvcc >/dev/null 2>&1; then
    _cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    for _d in "$_cuda_root/lib64" "$_cuda_root/targets/x86_64-linux/lib"; do
        [[ -d "$_d" ]] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
    done
    unset _cuda_root _d
fi

NUM_GPUS="${NUM_GPUS:-$("$PYTHON" -c "import config; print(len(config.WORKER_GPUS))")}"
AGG_DEVICE="cuda:0"
AGG_TCP_PORT="${AGG_TCP_PORT:-$("$PYTHON" -c "import config; print(config.AGGREGATOR_PORTS.get(0, 50056))")}"
AGG_HEALTH_PORT="${AGG_HEALTH_PORT:-$("$PYTHON" -c "import config; print(config.AGGREGATOR_HEALTH_PORTS.get(0, 8000))")}"
AGG_PORT_MAP=$("$PYTHON" -c "import config; print(','.join(f'{i}:{config.AGGREGATOR_PORTS.get(i, 50056+i)}' for i in range($NUM_GPUS)))")
AGG_HEALTH_PORT_MAP=$("$PYTHON" -c "import config; print(','.join(f'{i}:{config.AGGREGATOR_HEALTH_PORTS.get(i, 8000+i)}' for i in range($NUM_GPUS)))")
CTRL_PORT="${CTRL_PORT:-8344}"
SCHEDULER="${SCHEDULER:-lorant}"
ADAPTER_PREFIX="${ADAPTER_PREFIX:-../sim-adapters/pool-10-r16/lora-}"
AGG_HEALTH_TIMEOUT=180
CTRL_HEALTH_TIMEOUT=180

_SCRIPT_NAME="$(basename "$0" .sh)"
log() { echo "[$_SCRIPT_NAME] $(date '+%H:%M:%S') $*"; }

kill_component() {
    local name=$1 pattern=$2
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        sleep 1
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
        log "$name stopped"
    fi
}

wait_for_health() {
    local url=$1 name=$2 max_wait=$3
    local start=$SECONDS
    while (( SECONDS - start < max_wait )); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log "$name healthy ($((SECONDS - start))s)"
            return 0
        fi
        sleep 1
    done
    log "ERROR: $name not healthy after ${max_wait}s"
    return 1
}

stop_services() {
    kill_component "Benchmark"        "controller.benchmark\|cold_start\|ipc_overhead\|throughput_scaling\|baseline_tpot"
    kill_component "Controller"       "controller.controller"
    kill_component "Template process" "template_process"
    kill_component "Workers"          "worker_sync"
    kill_component "Aggregator"       "aggregator"
    sleep 2
    local gpu_pids mps_pid
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u) || true
    mps_pid=$(pgrep -f "nvidia-cuda-mps-server" 2>/dev/null || true)
    for pid in $gpu_pids; do
        [[ "$pid" == "$mps_pid" ]] && continue
        ps -p "$pid" -o comm= 2>/dev/null | grep -q python && kill -9 "$pid" 2>/dev/null || true
    done
}

start_mps() {
    bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true
    sleep 1
    bash "$SCRIPT_DIR/setup_mps.sh" start
}

# ═══════════════════════════════════════════════════════════════════════════════
# Trace-replay runner: starts aggregator + controller, replays traces, reports.
# Used by run_trace_driven.sh and run_throughput.sh.
# Caller must set: TRACES (array), TRACES_DIR, RESULTS_DIR, TRACE_SUITE
# ═══════════════════════════════════════════════════════════════════════════════

_start_trace_services() {
    local agg_gpu_idx="${AGG_DEVICE#cuda:}"
    local agg_mps_pipe="/tmp/mps_${agg_gpu_idx}"

    log "Starting aggregator on $AGG_DEVICE..."
    nohup env CUDA_VISIBLE_DEVICES=0 \
              CUDA_MPS_PIPE_DIRECTORY="$agg_mps_pipe" \
        "$PYTHON" src/aggregator.py \
        --device "cuda:0" \
        --port "$AGG_TCP_PORT" \
        --health-port "$AGG_HEALTH_PORT" \
        > "$LOG_DIR/aggregator_0.log" 2>&1 &

    wait_for_health "http://localhost:${AGG_HEALTH_PORT}/health" "Aggregator" "$AGG_HEALTH_TIMEOUT"

    # For large prewarm, launch all GPU aggregators upfront via P2P copy
    local total_prewarm="${PREWARM_WORKERS:-20}"
    if [[ "${PREWARM:-0}" == "1" && $total_prewarm -gt 30 ]]; then
        log "Large prewarm ($total_prewarm) — launching $NUM_GPUS aggregators..."
        for gpu_idx in $(seq 1 $((NUM_GPUS - 1))); do
            local port=$("$PYTHON" -c "import config; print(config.AGGREGATOR_PORTS.get($gpu_idx, 50056+$gpu_idx))")
            local hport=$("$PYTHON" -c "import config; print(config.AGGREGATOR_HEALTH_PORTS.get($gpu_idx, 8000+$gpu_idx))")
            nohup env CUDA_VISIBLE_DEVICES=0 \
                      CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_${gpu_idx}" \
                "$PYTHON" src/aggregator.py \
                --device "cuda:0" --port "$port" --health-port "$hport" \
                --donor-host "localhost" --donor-port "$AGG_TCP_PORT" \
                > "$LOG_DIR/aggregator_${gpu_idx}.log" 2>&1 &
        done
        for gpu_idx in $(seq 1 $((NUM_GPUS - 1))); do
            local hport=$("$PYTHON" -c "import config; print(config.AGGREGATOR_HEALTH_PORTS.get($gpu_idx, 8000+$gpu_idx))")
            wait_for_health "http://localhost:${hport}/health" "Aggregator-$gpu_idx" "$AGG_HEALTH_TIMEOUT"
        done
    fi

    local scale_down="${SCALE_DOWN_DELAY:-20}"
    [[ "${PREWARM:-0}" == "1" && $total_prewarm -gt 30 ]] && scale_down="${SCALE_DOWN_DELAY:-300}"

    # Pool-lifecycle trace: timestamped spawn/swap/reap events with running pool
    # size, so worker-count-over-time can be reconstructed after a run. Written
    # server-side; the client API deliberately omits worker identities.
    # Write the pool-event trace into RESULTS_DIR so it is retained with the
    # run it describes, keeping spawn/reap timing available for later analysis.
    # Falls back to logs/ when RESULTS_DIR is unset.
    local event_log="${SWARM_EVENT_LOG:-${RESULTS_DIR:-$LOG_DIR}/pool_events.jsonl}"
    mkdir -p "$(dirname "$event_log")"

    # The per-GPU worker cap comes from config.py, which honours the
    # MAX_WORKERS_PER_GPU env var, so the controller's spawn cap and every
    # aggregator's slot pool derive from the same value.
    if [[ -n "${MAX_WORKERS_PER_GPU:-}" ]]; then
        log "Per-GPU worker cap = $MAX_WORKERS_PER_GPU (env; applies to slots too)"
    fi

    log "Starting controller on port $CTRL_PORT (scheduler=$SCHEDULER)..."
    SWARM_EVENT_LOG="$event_log" nohup "$PYTHON" -m controller.controller \
        --port "$CTRL_PORT" \
        --aggregator-port-map "$AGG_PORT_MAP" \
        --aggregator-health-port-map "$AGG_HEALTH_PORT_MAP" \
        --scheduler "$SCHEDULER" \
        --scale-down-delay "$scale_down" \
        > "$LOG_DIR/controller.log" 2>&1 &

    wait_for_health "http://localhost:${CTRL_PORT}/health" "Controller" "$CTRL_HEALTH_TIMEOUT"
}

_prewarm() {
    [[ "${PREWARM:-0}" != "1" ]] && return 0
    local total="${PREWARM_WORKERS:-20}"
    local adapter="${PREWARM_ADAPTER:-${ADAPTER_PREFIX}4}"
    log "Prewarming $total workers with $adapter..."
    curl -sf -X POST "http://localhost:${CTRL_PORT}/prewarm" \
        -H "Content-Type: application/json" \
        -d "{\"adapters\": {\"${adapter}\": ${total}}}" \
        --max-time 300 >/dev/null || log "WARN: Prewarm failed"
}

run_traces() {
    trap 'log "Interrupted!"; stop_services; bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true; exit 130' INT TERM

    local start_time=$SECONDS

    log "============================================================"
    log "TRACE REPLAY: ${#TRACES[@]} traces from $TRACES_DIR"
    log "Results: $RESULTS_DIR"
    log "============================================================"

    for trace in "${TRACES[@]}"; do
        [[ ! -f "$TRACES_DIR/${trace}.jsonl" ]] && { log "ERROR: Missing $TRACES_DIR/${trace}.jsonl"; exit 1; }
    done

    stop_services
    rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.sock 2>/dev/null
    start_mps

    # Run each trace with clean restart
    local idx=0
    for trace in "${TRACES[@]}"; do
        idx=$((idx + 1))
        local trace_file="$TRACES_DIR/${trace}.jsonl"
        local trace_name="${trace%%_${TRACE_SUITE}}"
        local trace_dir="$RESULTS_DIR/$trace_name"
        mkdir -p "$trace_dir"

        log "[$idx/${#TRACES[@]}] $trace_name ($(wc -l < "$trace_file") requests)"

        stop_services
        rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.sock 2>/dev/null
        _start_trace_services
        _prewarm

        "$PYTHON" -m controller.benchmark \
            --api-url "http://localhost:${CTRL_PORT}" \
            --trace-file "$trace_file" \
            --adapter-map-prefix "$ADAPTER_PREFIX" \
            --output-dir "$trace_dir"

        local latest=$(ls -t "$trace_dir"/summary_*.json 2>/dev/null | head -1)
        [[ -n "$latest" ]] && cp "$latest" "$trace_dir/summary.json"
        local latest_records=$(ls -t "$trace_dir"/records_*.json 2>/dev/null | head -1)
        [[ -n "$latest_records" ]] && cp "$latest_records" "$trace_dir/records.json"
    done

    # Cleanup
    stop_services
    bash "$SCRIPT_DIR/setup_mps.sh" stop 2>/dev/null || true

    local elapsed=$(( SECONDS - start_time ))
    log "Done in $((elapsed / 60))m $((elapsed % 60))s. Results: $RESULTS_DIR"
}
