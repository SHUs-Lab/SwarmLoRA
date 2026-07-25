#!/usr/bin/env bash
# Security isolation: malicious-adapter data theft + fault-injection blast radius.
# Reproduces both columns of Table III: S-LoRA (serverful baseline) vs. SwarmLoRA.
#
# Usage: bash scripts/run_security.sh [all|swarmlora-only]
#
# The SwarmLoRA-side experiments are self-contained. The S-LoRA side uses the
# bundled baseline at baselines/slora (one-time venv build, see its README) --
# set SLORA=/path/to/checkout to use a different one, or pass swarmlora-only.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEC="$ROOT/security_eval"
SLORA="${SLORA:-$ROOT/baselines/slora}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmark_results/security}"
mkdir -p "$RESULTS_DIR"

# Host CUDA can be a different major version than the pinned torch wheels' bundled
# runtime, so the SwarmLoRA/S-LoRA extensions may need the host's libcudart.so on
# the loader path. Exported here so every sub-experiment's python child inherits it.
if command -v nvcc >/dev/null 2>&1; then
    _cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    for _d in "$_cuda_root/lib64" "$_cuda_root/targets/x86_64-linux/lib"; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
    done
    unset _cuda_root _d
fi

MODE="${1:-all}"   # all | swarmlora-only

ok=0; fail=0
run() {
    local label="$1" script="$2" out="$3"
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  RUNNING: $label"
    echo "══════════════════════════════════════════════════"
    if bash "$script" "$out"; then
        echo "  PASSED: $label -> $out"
        ok=$((ok + 1))
    else
        echo "  FAILED: $label"
        fail=$((fail + 1))
    fi
}

run "blind_attack_swarmlora (Table III: weight theft, KV/activation leakage)" \
    "$SEC/malicious_adapter/swarmlora/blind_attack_swarmlora.sh" \
    "$RESULTS_DIR/blind_attack_swarmlora.json"

run "fault_swarmlora (Table III: fault blast radius, SIGKILL/SIGSEGV/SIGABRT)" \
    "$SEC/fault/swarmlora/runner.sh" \
    "$RESULTS_DIR/fault_swarmlora.json"

if [[ "$MODE" != "swarmlora-only" ]]; then
    if [[ -x "$SLORA/venv/bin/python" ]]; then
        SLORA="$SLORA" run "blind_attack_slora (Table III baseline: weight theft, KV/activation leakage)" \
            "$SEC/malicious_adapter/slora/blind_attack_slora.sh" \
            "$RESULTS_DIR/blind_attack_slora.json"

        SLORA="$SLORA" run "fault_slora (Table III baseline: fault blast radius)" \
            "$SEC/fault/slora/runner.sh" \
            "$RESULTS_DIR/fault_slora.json"
    else
        echo ""
        echo "[run_security] $SLORA/venv not built — skipping S-LoRA baseline experiments."
        echo "[run_security] One-time setup: bash $SLORA/setup.sh"
        echo "[run_security] Or pass 'swarmlora-only' to silence this message."
    fi
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  SECURITY SUITE: passed=$ok failed=$fail"
echo "  Results: $RESULTS_DIR"
echo "══════════════════════════════════════════════════"

# Render Table III if all four result JSONs are present; silent otherwise
# (swarmlora-only, or the S-LoRA venv isn't built).
if [[ -f "$RESULTS_DIR/blind_attack_slora.json" && -f "$RESULTS_DIR/fault_slora.json" \
   && -f "$RESULTS_DIR/blind_attack_swarmlora.json" && -f "$RESULTS_DIR/fault_swarmlora.json" ]]; then
    _py="$ROOT/venv/bin/python"; [[ -x "$_py" ]] || _py=python3
    "$_py" "$ROOT/analysis/security_table.py" || true
fi

[[ "$fail" -eq 0 ]]
