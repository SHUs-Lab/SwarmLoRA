#!/bin/bash
# Run the Poisson throughput sweep (Fig. 5 comparison point).
# Usage: bash scripts/run_throughput.sh
exec bash "$(dirname "$0")/run_all_traces.sh" --duration 3min --trace-dir ../../traces/3min_poisson "$@"
