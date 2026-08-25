#!/usr/bin/env bash
# Throughput under load: Poisson arrivals at 1-12 RPS (Fig 5).
# Usage: bash scripts/run_throughput.sh
#   env override: RESULTS_DIR (default: ./benchmark_results/throughput)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM
trap stop_services EXIT

export PREWARM=1 PREWARM_WORKERS=20
# A request counts as failed once it can no longer meet its latency target.
export ADMISSION_SLO_S="${ADMISSION_SLO_S:-6}"

TRACE_SUITE=3min_10adp
TRACES_DIR="traces/3min_poisson"
RESULTS_DIR="${RESULTS_DIR:-./benchmark_results/throughput}"
# Optional rate selection, mirroring run_trace_driven.sh: pass bare rates to
# run a subset, e.g. `bash scripts/run_throughput.sh 8` or `... 4 8 12`.
# Default remains the full sweep used for Fig 5.
if [[ $# -gt 0 ]]; then
    TRACES=()
    for rate in "$@"; do
        TRACES+=("poisson_rps${rate}_${TRACE_SUITE}")
    done
else
    TRACES=(
        "poisson_rps1_${TRACE_SUITE}"
        "poisson_rps2_${TRACE_SUITE}"
        "poisson_rps4_${TRACE_SUITE}"
        "poisson_rps8_${TRACE_SUITE}"
        "poisson_rps12_${TRACE_SUITE}"
    )
fi

run_traces
