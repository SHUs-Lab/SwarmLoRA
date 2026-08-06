#!/usr/bin/env bash
# Measure the "Standard Load" cold-start baseline (Fig. 6) -- plain HF
# from_pretrained with no optimization, for comparison against SwarmLoRA,
# ServerlessLoRA, and ServerlessLLM's cold-start numbers.
# Usage: bash scripts/run_standard_load_cold_start.sh [num_runs]
#
# Needs no aggregator/controller/MPS, so it doesn't source common.sh -- hence
# its own strict mode rather than the inherited one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_RUNS="${1:-4}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmark_results/cold_start}"
mkdir -p "$RESULTS_DIR"

echo "[$(date '+%H:%M:%S')] Running standard-load cold-start benchmark (num_runs=$NUM_RUNS)..."
venv/bin/python benchmarks/standard_load_cold_start.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --adapter-path ../sim-adapters/pool-10-r16/lora-0 \
    --num-runs "$NUM_RUNS" \
    --output "$RESULTS_DIR/standard_load_cold_start.json"

echo "[$(date '+%H:%M:%S')] Done."
