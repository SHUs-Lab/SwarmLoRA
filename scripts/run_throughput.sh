#!/usr/bin/env bash
# Throughput under load: Poisson arrivals at 1-12 RPS (Fig 5).
# Usage: bash scripts/run_throughput.sh
#   env override: RESULTS_DIR (default: ./benchmark_results/throughput)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM
trap stop_services EXIT

export PREWARM=1 PREWARM_WORKERS=20
# Matches the paper's acceptance metric: requests queued >8s count as failed.
export QUEUE_TIMEOUT="${QUEUE_TIMEOUT:-8}"

TRACE_SUITE=3min_10adp
TRACES_DIR="traces/3min_poisson"
RESULTS_DIR="${RESULTS_DIR:-./benchmark_results/throughput}"
TRACES=(
    "poisson_rps1_${TRACE_SUITE}"
    "poisson_rps2_${TRACE_SUITE}"
    "poisson_rps4_${TRACE_SUITE}"
    "poisson_rps8_${TRACE_SUITE}"
    "poisson_rps12_${TRACE_SUITE}"
)

run_traces
