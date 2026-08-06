#!/usr/bin/env bash
# One-shot environment setup for SwarmLoRA artifact evaluation.
# Usage: bash scripts/setup.sh
#
# Time estimate: ~20 min total
#   - Prerequisite checks:               <1 min
#   - venv + pip install:                ~8 min
#   - Build C++/CUDA extensions:         ~3 min
#   - Adapter generation (7 pools):      ~8 min
# Excludes: one-time Llama-3.1-8B download (~16 GB, depends on network speed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

log() { echo "[setup] $(date '+%H:%M:%S') $*"; }
fail() { echo "[setup] ERROR: $*" >&2; exit 1; }

START=$SECONDS

# ── Step 0: Prerequisite checks ──────────────────────────────────────────────
log "Checking prerequisites..."

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — an NVIDIA GPU with drivers installed is required."
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
[[ "$NUM_GPUS" -ge 1 ]] || fail "No GPUs detected by nvidia-smi."
log "Found $NUM_GPUS GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/[setup]   /'

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
if [[ "$VRAM_MB" -lt 24000 ]]; then
    log "WARNING: GPU 0 has ${VRAM_MB} MiB VRAM (<24 GB recommended for Llama-3.1-8B + workers)."
fi

command -v gcc >/dev/null 2>&1 || fail "gcc not found (GCC 11+ required for C++17 extension build)."
command -v g++ >/dev/null 2>&1 || fail "g++ not found (GCC 11+ required for C++17 extension build)."
GCC_VER=$(gcc -dumpversion | cut -d. -f1)
[[ "$GCC_VER" -ge 11 ]] || log "WARNING: gcc version $GCC_VER detected, 11+ recommended."

command -v nvcc >/dev/null 2>&1 || log "WARNING: nvcc not found on PATH — CUDA toolkit may not be installed/loaded (needed for extension build)."

# Pinned dependencies (torch==2.5.1 etc.) are validated against Python 3.9 --
# plain `python3` silently resolves to whatever the OS ships (3.10 on Ubuntu
# 22.04, 3.12 on 24.04, ...), so require 3.9 specifically rather than risk a
# version-skewed build no one asked for.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    : # explicit override, trust the caller
elif command -v python3.9 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.9)"
elif command -v uv >/dev/null 2>&1; then
    uv python install 3.9
    PYTHON_BIN="$(uv python find 3.9)"
else
    fail "No python3.9 on PATH and uv not found. Install Python 3.9 (e.g. via deadsnakes on Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.9 python3.9-venv) or uv (https://docs.astral.sh/uv/), then re-run."
fi
PY_VER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
log "Python: $PY_VER ($(command -v "$PYTHON_BIN"))"

log "Prerequisites OK ($((SECONDS - START))s)"

# ── Step 1: Virtual environment + dependencies ───────────────────────────────
if [[ ! -d "venv" ]]; then
    log "Creating venv..."
    "$PYTHON_BIN" -m venv venv
else
    log "venv already exists, reusing."
fi

log "Installing Python dependencies (this may take several minutes)..."
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt
log "Dependencies installed ($((SECONDS - START))s elapsed)"

# ── Step 2: Build C++/CUDA extensions ────────────────────────────────────────
log "Building C++/CUDA extensions (ext_aggregator, ext_ipc_queue, ext_unified_barrier)..."
venv/bin/python setup.py build_ext --inplace
log "Extensions built ($((SECONDS - START))s elapsed)"

# ── Step 2.5: Reconcile HF_HOME with the standard cache location ────────────
# Some platforms default HF_HOME to a non-standard path. If model data lands at
# the standard ~/.cache/huggingface while HF_HOME points elsewhere,
# local_files_only lookups silently miss it. Symlink so both paths agree.
_std_hf_home="$HOME/.cache/huggingface"
_env_hf_home="${HF_HOME:-$_std_hf_home}"
if [[ "$_env_hf_home" != "$_std_hf_home" ]]; then
    if [[ -d "$_std_hf_home/hub" && ! -e "$_env_hf_home/hub" ]]; then
        mkdir -p "$_env_hf_home"
        ln -sfn "$_std_hf_home/hub" "$_env_hf_home/hub"
        log "Linked HF cache: $_env_hf_home/hub -> $_std_hf_home/hub"
    elif [[ -d "$_env_hf_home/hub" && ! -e "$_std_hf_home/hub" ]]; then
        mkdir -p "$_std_hf_home"
        ln -sfn "$_env_hf_home/hub" "$_std_hf_home/hub"
        log "Linked HF cache: $_std_hf_home/hub -> $_env_hf_home/hub"
    fi
fi
unset _std_hf_home _env_hf_home

# ── Step 3: HuggingFace authentication ───────────────────────────────────────
# Skip entirely if the model is already cached locally -- inference will use
# the cache and never touch the network, so there's nothing to authenticate.
if venv/bin/python -c "
from huggingface_hub import scan_cache_dir
import sys
sys.exit(0 if any(r.repo_id == 'meta-llama/Llama-3.1-8B-Instruct' for r in scan_cache_dir().repos) else 1)
" >/dev/null 2>&1; then
    log "Llama-3.1-8B-Instruct already cached — skipping HuggingFace auth."
elif [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN set — logging in non-interactively..."
    venv/bin/huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
elif venv/bin/python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
    log "Already logged in to HuggingFace."
else
    log "WARNING: Not logged in to HuggingFace and HF_TOKEN not set."
    log "         Llama-3.1-8B-Instruct requires a HuggingFace account with model access granted."
    log "         Run 'venv/bin/huggingface-cli login' before running any RQ script, or set HF_TOKEN and re-run this script."
fi

# ── Step 3.5: Ensure model + tokenizer are BOTH fully cached ─────────────────
# Workers load with local_files_only=True and never fetch anything themselves,
# so a partial cache (weights but no tokenizer, or vice versa) fails deep in a
# benchmark run instead of here. Idempotent.
log "Ensuring Llama-3.1-8B-Instruct (weights + tokenizer) is fully cached..."
venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
" || fail "Could not fetch meta-llama/Llama-3.1-8B-Instruct (weights or tokenizer). Check HF_TOKEN / model access, then re-run scripts/setup.sh."
log "Model + tokenizer cache verified ($((SECONDS - START))s elapsed)"

# ── Step 4: Per-GPU MPS daemons ──────────────────────────────────────────────
log "Starting per-GPU MPS daemons..."
bash "$SCRIPT_DIR/setup_mps.sh" start

# ── Step 5: Generate adapter pools ───────────────────────────────────────────
# rank-16 pool for RQ1 (throughput), RQ3 (trace-driven), RQ5 (overhead);
# mixed-rank pools for RQ4 (scalability sweep, 10-500 adapters).
# Skip pools that already exist -- generate_adapters.py always rebuilds from
# scratch, so a re-run of this script would otherwise redo ~8 min of work.
if [[ -d "../sim-adapters/pool-10-r16/lora-0" ]]; then
    log "Adapter pool-10-r16 already exists, skipping."
else
    log "Generating rank-16 adapter pool (10 adapters)..."
    venv/bin/python generate_adapters.py --rank 16 --num 10 --out ../sim-adapters/pool-10-r16
fi
for N in 10 20 50 100 200 500; do
    if [[ -e "../sim-adapters/pool-$N" ]]; then
        log "Adapter pool-$N already exists, skipping."
    else
        log "Generating mixed-rank adapter pool ($N adapters)..."
        venv/bin/python generate_adapters.py --mixed --num "$N" --symlink --out "../sim-adapters/pool-$N"
    fi
done
log "Adapter pools ready ($((SECONDS - START))s elapsed)"

# ── Done ──────────────────────────────────────────────────────────────────────
ELAPSED=$((SECONDS - START))
log "============================================================"
log "SETUP COMPLETE in $((ELAPSED / 60))m $((ELAPSED % 60))s"
log "============================================================"
log "Next: bash scripts/run_throughput.sh   (or any other scripts/run_*.sh)"
