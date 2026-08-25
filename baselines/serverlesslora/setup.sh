#!/usr/bin/env bash
# Usage: bash baselines/serverlesslora/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[setup-serverlesslora] $*"; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "[setup-serverlesslora] ERROR: nvidia-smi not found." >&2; exit 1; }

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

if [[ ! -d "sless-venv" ]]; then
    log "Creating venv"
    python3 -m venv sless-venv
else
    log "venv already exists, reusing."
fi

log "Installing dependencies"
sless-venv/bin/pip install -r requirements.txt

log "Building extensions"
sless-venv/bin/python setup.py build_ext --inplace

# Host CUDA can be a different major version than torch's bundled runtime, so the
# extensions may need the host's libcudart.so. Appended to the venv's activate so
# every run script picks it up automatically.
if command -v nvcc >/dev/null 2>&1 && ! grep -q "SLESS_CUDA_LIBPATH" sless-venv/bin/activate; then
    log "Registering host CUDA runtime libs on the venv"
    cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    echo "# SLESS_CUDA_LIBPATH" >> sless-venv/bin/activate
    for d in "$cuda_root/lib64" "$cuda_root/targets/x86_64-linux/lib"; do
        [ -d "$d" ] && echo "export LD_LIBRARY_PATH=\"$d:\${LD_LIBRARY_PATH:-}\"" >> sless-venv/bin/activate
    done
fi

log "Done"
log "Next: bash baselines/serverlesslora/scripts/run_throughput.sh   (or any other scripts/run_*.sh)"
