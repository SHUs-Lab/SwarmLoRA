#!/usr/bin/env bash
# Trace-driven evaluation: 5 traffic categories x 10 minutes (Figs 6-7,9-11, Table II).
# Usage: bash scripts/run_trace_driven.sh [trace_name ...]
#   trace_name: one or more of steady_light bursty_light normal steady_heavy
#               bursty_heavy (default: all five)
#   env override: RESULTS_DIR (default: ./benchmark_results/10min_lorant_pw20)
source "$(dirname "$0")/common.sh"
trap 'stop_services; exit 130' INT TERM
trap stop_services EXIT

export PREWARM=1 PREWARM_WORKERS=20
export SCALE_DOWN_DELAY="${SCALE_DOWN_DELAY:-40}"
# A request counts as failed once it can no longer meet its latency target.
export ADMISSION_SLO_S="${ADMISSION_SLO_S:-6}"

TRACE_SUITE=10min_10adp
TRACES_DIR="traces/${TRACE_SUITE}"
RESULTS_DIR="${RESULTS_DIR:-./benchmark_results/10min_lorant_pw20}"
if [[ $# -gt 0 ]]; then
    TRACES=("$@")
    TRACES=("${TRACES[@]/%/_${TRACE_SUITE}}")
else
    TRACES=(
        "bursty_heavy_${TRACE_SUITE}"
        "steady_heavy_${TRACE_SUITE}"
        "normal_${TRACE_SUITE}"
        "bursty_light_${TRACE_SUITE}"
        "steady_light_${TRACE_SUITE}"
    )
fi

run_traces
