#!/usr/bin/env bash
# Usage: bash baselines/serverlessllm/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[setup-serverlessllm] $*"; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "[setup-serverlessllm] ERROR: nvidia-smi not found." >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "[setup-serverlessllm] ERROR: cmake not found. Install it: sudo apt install cmake (Ubuntu/Debian), sudo dnf install cmake (Fedora/RHEL), or see https://cmake.org/download/" >&2; exit 1; }

# Some platforms default HF_HOME to a non-standard path. If model data lands at
# the standard ~/.cache/huggingface while HF_HOME points elsewhere,
# local_files_only lookups silently miss it. Symlink so both paths agree.
_std_hf_home="$HOME/.cache/huggingface"
_env_hf_home="${HF_HOME:-$_std_hf_home}"
if [[ "$_env_hf_home" != "$_std_hf_home" ]]; then
    if [[ -d "$_std_hf_home/hub" && ! -e "$_env_hf_home/hub" ]]; then
        mkdir -p "$_env_hf_home"; ln -sfn "$_std_hf_home/hub" "$_env_hf_home/hub"
        log "Linked HF cache: $_env_hf_home/hub -> $_std_hf_home/hub"
    elif [[ -d "$_env_hf_home/hub" && ! -e "$_std_hf_home/hub" ]]; then
        mkdir -p "$_std_hf_home"; ln -sfn "$_env_hf_home/hub" "$_std_hf_home/hub"
        log "Linked HF cache: $_std_hf_home/hub -> $_env_hf_home/hub"
    fi
fi
unset _std_hf_home _env_hf_home

if [[ ! -d "sllm-env" ]]; then
    log "Creating venv"
    python3 -m venv sllm-env
else
    log "venv already exists, reusing."
fi

log "Upgrading pip/setuptools/wheel"
sllm-env/bin/pip install --upgrade pip setuptools wheel

log "Installing dependencies"
sllm-env/bin/pip install --no-build-isolation -r requirements.txt -r sllm_store/requirements.txt
sllm-env/bin/pip install "datasets>=3.6.0"

log "Building sllm_store"
cd sllm_store
../sllm-env/bin/python setup.py build_ext --inplace
../sllm-env/bin/python setup.py develop
cd ..

log "Installing libglog"
cp -P sllm_store/build/lib.linux-x86_64-*/sllm_store/libglog.so* sllm_store/sllm_store/

log "Installing serverless-llm"
sllm-env/bin/python setup.py develop

# Link the shared adapter pool at models/raw_adapters/adapter_{i} (raw
# pre-conversion PEFT format) so all three systems use identical weights.
log "Linking shared adapter pool"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAPTER_POOL="$(dirname "$REPO_ROOT")/sim-adapters/pool-10-r16"
mkdir -p models/raw_adapters
for i in 0 1 2 3 4 5 6 7 8 9; do
    ln -sfn "$ADAPTER_POOL/lora-$i" "models/raw_adapters/adapter_$i"
done

# The server's own registration path random-initializes an 8B skeleton per
# adapter on CPU (~5 min each); fast_register_adapters.py loads the real base
# model once instead, taking seconds. The server's "already exists" check then
# skips registration at RQ1/RQ3 launch. See that script for details.
log "Pre-registering model + adapters into sllm_store format (fast path)..."
sllm-env/bin/python scripts/fast_register_adapters.py --storage-path ./models

# An interrupted conversion can leave a valid-looking tensor_index.json beside
# a truncated tensor.data_0; the "already exists" check can't detect that and
# would silently serve corrupt data. Drop anything truncated so it reconverts.
log "Verifying any pre-existing converted adapters are complete..."
for i in 0 1 2 3 4 5 6 7 8 9; do
    dir="models/transformers/adapter_$i"
    [[ -f "$dir/tensor_index.json" && -f "$dir/tensor.data_0" ]] || continue
    ok=$(python3 -c "
import json, os
idx = json.load(open('$dir/tensor_index.json'))
max_end = max(off + sz for off, sz, *_ in idx.values())
actual = os.path.getsize('$dir/tensor.data_0')
print('ok' if actual >= max_end else 'truncated')
" 2>/dev/null || echo "truncated")
    if [[ "$ok" != "ok" ]]; then
        log "  $dir: truncated/corrupt data detected, deleting -- will be regenerated"
        rm -rf "$dir"
    fi
done

log "Done"
log "Next: bash baselines/serverlessllm/scripts/run_throughput.sh   (or any other scripts/run_*.sh)"
