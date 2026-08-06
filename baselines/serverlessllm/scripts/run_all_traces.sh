#!/bin/bash
# Run all trace benchmarks with clean server restart between each trace.
#
# Usage: bash scripts/run_all_traces.sh --duration 10min
#   (or use the wrapper: bash scripts/run_throughput.sh / run_trace_driven.sh)
#
# Prerequisites: models already downloaded in ./models/

set -e
cd "$(dirname "$0")/.."
source sllm-env/bin/activate

# Ray sizes its worker pool by CPU count, and each worker consumes several file
# descriptors. On many-core hosts the default 1024 soft limit is exhausted
# during `ray start` and the raylet aborts with "Too many open files" (SIGABRT),
# which surfaces later as "Unable to register worker with raylet".
FD_TARGET=65535
FD_HARD=$(ulimit -Hn)
if [ "$FD_HARD" != "unlimited" ] && [ "$FD_HARD" -lt "$FD_TARGET" ]; then
    FD_TARGET="$FD_HARD"
fi
ulimit -n "$FD_TARGET" 2>/dev/null \
    || echo "[warn] could not raise FD limit (soft=$(ulimit -Sn)); ray may abort on many-core hosts"

# Quiet sllm's own DEBUG/INFO chatter; server stdout already goes to
# logs/sllm_server.log below.
export LOG_LEVEL="${LOG_LEVEL:-ERROR}"

# Same latency target as the other systems (src/controller/admission.py).
export ADMISSION_SLO_S="${ADMISSION_SLO_S:-6}"

# --- Auto-detect GPUs ---
NUM_GPUS_TOTAL=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "[auto] Detected $NUM_GPUS_TOTAL GPU(s)"

# --- Configuration ---
# 10min matches the trace suite RQ3 uses; run_throughput.sh overrides it with
# --duration 3min --trace-dir for the Poisson sweep.
DURATION="10min"
MODE="optimized"
PREFIX="sllm_optimized"
INSTANCES_PER_GPU=2
NUM_ADAPTERS=10
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# Absolute: Ray workers may run with a different cwd, so a relative path makes
# their already-exists check miss real files and redo full registration.
STORAGE_PATH="$(pwd)/models"
MEM_POOL_SIZE="20GB"
OUTPUT_DIR="benchmark_results/trace_driven"
SERVER_PORT=8343
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
SETUP_TIMEOUT=600
REQUEST_TIMEOUT=600

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --instances-per-gpu) INSTANCES_PER_GPU="$2"; shift 2 ;;
        --num-adapters) NUM_ADAPTERS="$2"; shift 2 ;;
        --mem-pool-size) MEM_POOL_SIZE="$2"; shift 2 ;;
        --storage-path) STORAGE_PATH="$2"; shift 2 ;;
        --timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
        --trace) SINGLE_TRACE="$2"; shift 2 ;;
        --trace-dir) TRACE_DIR_OVERRIDE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Resolve trace directory and file suffix ---
if [ -n "${TRACE_DIR_OVERRIDE:-}" ]; then
    TRACE_DIR="$TRACE_DIR_OVERRIDE"
    TRACE_SUFFIX=$(basename "$TRACE_DIR")
else
    TRACE_DIR="../../traces/${DURATION}_${NUM_ADAPTERS}adp"
    TRACE_SUFFIX="${DURATION}_${NUM_ADAPTERS}adp"
fi

if [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory not found: $TRACE_DIR"
    echo "Available:"
    ls -d ../../traces/*/ 2>/dev/null || echo "  (none)"
    exit 1
fi

# Ordered light-to-heavy for predictable benchmarking
TRACE_ORDER=(steady_light bursty_light normal steady_heavy bursty_heavy)
TRACES=()
for pattern in "${TRACE_ORDER[@]}"; do
    for f in "$TRACE_DIR"/${pattern}_*.jsonl; do
        if [ -f "$f" ]; then
            name=$(basename "$f" .jsonl)
            TRACES+=("$name")
        fi
    done
done

# If no ordered traces found (e.g. poisson), sort numerically by rps so the
# sweep runs light-to-heavy; lexicographic order would run rps12 before rps2.
if [ ${#TRACES[@]} -eq 0 ]; then
    while IFS=$'\t' read -r _ name; do
        TRACES+=("$name")
    done < <(
        for f in "$TRACE_DIR"/*.jsonl; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .jsonl)
            rate=$(echo "$name" | sed -E 's/.*rps([0-9.]+).*/\1/')
            printf '%s\t%s\n' "$rate" "$name"
        done | sort -t$'\t' -k1,1n
    )
fi

if [ ${#TRACES[@]} -eq 0 ]; then
    echo "ERROR: No .jsonl files found in $TRACE_DIR"
    exit 1
fi

# Filter to single trace if --trace specified
if [ -n "${SINGLE_TRACE:-}" ]; then
    FILTERED=()
    for t in "${TRACES[@]}"; do
        if [[ "$t" == *"$SINGLE_TRACE"* ]]; then
            FILTERED+=("$t")
        fi
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "ERROR: No trace matching '$SINGLE_TRACE'. Available: ${TRACES[*]}"
        exit 1
    fi
    TRACES=("${FILTERED[@]}")
fi

# For 10min traces, increase timeout
if [[ "$DURATION" == "10min" ]]; then
    REQUEST_TIMEOUT=1800
fi

# --- Compute derived values ---
NUM_GPU_PER_INSTANCE=$(python3 -c "print(round(1.0 / $INSTANCES_PER_GPU, 2))")
TOTAL_INSTANCES=$((NUM_GPUS_TOTAL * INSTANCES_PER_GPU))

# Build Ray worker_id resources
WORKER_IDS=""
for ((i=0; i<NUM_GPUS_TOTAL; i++)); do
    if [ $i -gt 0 ]; then WORKER_IDS+=","; fi
    WORKER_IDS+="\"worker_id_${i}\":1"
done
RAY_RESOURCES="{\"control_node\":1,\"worker_node\":1,${WORKER_IDS}}"

OUTPUT_SUFFIX="${TOTAL_INSTANCES}inst_${NUM_GPUS_TOTAL}gpu"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

cleanup() {
    echo "[cleanup] Killing server, sllm-store, and Ray..."
    pkill -f "benchmarks/start_server.py" 2>/dev/null || true
    pkill -f "sllm-store start" 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 2
    echo "[cleanup] Done"
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

start_services() {
    echo "[start] Starting NVIDIA MPS daemon..."
    nvidia-cuda-mps-control -d

    echo "[start] Starting Ray with $NUM_GPUS_TOTAL GPUs..."
    # PYTHONPATH must match start_server.py's below -- a mismatch double-imports
    # storage_pb2 and crashes Ray workers with a masked ActorDiedError.
    PYTHONPATH="sllm_store:.:$PYTHONPATH" ray start --head --num-gpus="$NUM_GPUS_TOTAL" --disable-usage-stats \
        --resources="$RAY_RESOURCES" 2>&1 | tail -1

    echo "[start] Starting sllm-store... (log: logs/sllm_store_server.log)"
    sllm-store start --storage-path "$STORAGE_PATH" --mem-pool-size "$MEM_POOL_SIZE" \
        > logs/sllm_store_server.log 2>&1 &
    sleep 5

    echo "[start] Starting server... (log: logs/sllm_server.log)"
    PYTHONPATH="sllm_store:.:$PYTHONPATH" STORAGE_PATH="$STORAGE_PATH" python3 -u benchmarks/start_server.py \
        > logs/sllm_server.log 2>&1 &

    echo "[start] Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s "$SERVER_URL/health" | grep -q "ok"; then
            echo "[start] Server is ready"
            return 0
        fi
        sleep 2
    done
    echo "[start] ERROR: Server failed to start within 120s"
    return 1
}

register_model() {
    echo "[setup] Registering model: $MODEL_NAME"
    echo "[setup]   mode=$MODE, gpu/inst=$NUM_GPU_PER_INSTANCE, instances=$TOTAL_INSTANCES, adapters=$NUM_ADAPTERS"
    PYTHONPATH="sllm_store:.:$PYTHONPATH" python3 benchmarks/setup_model.py \
        --mode "$MODE" \
        --num-gpus "$NUM_GPU_PER_INSTANCE" \
        --max-instances "$TOTAL_INSTANCES" \
        --min-instances "$TOTAL_INSTANCES" \
        --model-name "$MODEL_NAME" \
        --num-adapters "$NUM_ADAPTERS" \
        --timeout "$SETUP_TIMEOUT"
}

run_trace() {
    local trace_name="$1"
    local trace_file="${TRACE_DIR}/${trace_name}.jsonl"
    # Strip the duration+adapter suffix for cleaner output names
    local short_name="${trace_name%%_${TRACE_SUFFIX}}"
    local output_file="${OUTPUT_DIR}/${PREFIX}_${short_name}_${DURATION}_${OUTPUT_SUFFIX}.json"

    echo ""
    echo "========================================================"
    echo "  Trace:     $trace_name"
    echo "  Duration:  $DURATION"
    echo "  Instances: $TOTAL_INSTANCES ($INSTANCES_PER_GPU per GPU)"
    echo "  Output:    $output_file"
    echo "========================================================"

    # RQ1 (3min Poisson sweep) only needs throughput for Fig. 5;
    # RQ3 (10min trace-driven) needs TTFT/TPOT too for Table III/Figs. 7-9.
    local fields_arg="all"
    [[ "$DURATION" == "3min" ]] && fields_arg="throughput"

    PYTHONPATH="sllm_store:.:$PYTHONPATH" python3 benchmarks/replay_trace.py \
        --trace-file "$trace_file" \
        --output-file "$output_file" \
        --mode "$MODE" \
        --timeout-per-request "$REQUEST_TIMEOUT" \
        --fields "$fields_arg"
}

# --- Main ---
echo "=============================================="
echo "  ServerlessLLM Trace Benchmark Suite"
echo "=============================================="
echo "  Duration:          $DURATION"
echo "  Trace dir:         $TRACE_DIR"
echo "  Mode:              $MODE"
echo "  Prefix:            $PREFIX"
echo "  GPUs detected:     $NUM_GPUS_TOTAL"
echo "  Instances/GPU:     $INSTANCES_PER_GPU"
echo "  GPU/instance:      $NUM_GPU_PER_INSTANCE"
echo "  Total instances:   $TOTAL_INSTANCES"
echo "  Adapters:          $NUM_ADAPTERS"
echo "  Traces:            ${#TRACES[@]} (${TRACES[*]})"
echo "  Request timeout:   ${REQUEST_TIMEOUT}s"
echo "  Output suffix:     $OUTPUT_SUFFIX"
echo "=============================================="
echo ""

for trace_name in "${TRACES[@]}"; do
    echo ""
    echo ">>> Clean restart for: $trace_name"

    cleanup
    start_services
    register_model

    echo "[wait] Waiting 10s for instances to warm up..."
    sleep 10

    run_trace "$trace_name"

    echo ""
    echo ">>> Completed: $trace_name"
done

cleanup

echo ""
echo "=============================================="
echo "  All $DURATION traces complete!"
echo "  Results in: $OUTPUT_DIR/"
echo "=============================================="
