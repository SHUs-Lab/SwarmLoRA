#!/bin/bash
# Run all 10-minute trace benchmarks (Figs. 6-7, 9-11, Table II comparison point).
# Matches SwarmLoRA RQ3 and ServerlessLoRA, which both use the 10min_10adp traces.
# Usage: bash scripts/run_trace_driven.sh
exec bash "$(dirname "$0")/run_all_traces.sh" --duration 10min "$@"
