#!/usr/bin/env bash
# Usage: bash baselines/slora/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[setup-slora] $*"; }
fail() { echo "[setup-slora] ERROR: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found."

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

if [[ ! -d "venv" ]]; then
    if command -v python3.9 >/dev/null 2>&1; then
        PY39=$(command -v python3.9)
    elif command -v uv >/dev/null 2>&1; then
        uv python install 3.9
        PY39=$(uv python find 3.9)
    else
        fail "No python3.9 on PATH and uv not found. Install uv (https://docs.astral.sh/uv/) or a Python 3.9 interpreter, then re-run."
    fi
    log "Creating venv"
    "$PY39" -m venv venv
else
    log "venv already exists, reusing."
fi

log "Installing setuptools/wheel"
venv/bin/pip install "setuptools<81" wheel

log "Installing torch"
venv/bin/pip install torch==2.0.1

python3 - <<'PY'
f = "venv/lib/python3.9/site-packages/torch/utils/cpp_extension.py"
content = open(f).read()
old = "            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
new = "            warnings.warn(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
if old in content:
    open(f, "w").write(content.replace(old, new))
PY

log "Building S-LoRA"
venv/bin/pip install --no-build-isolation -e .

log "Installing triton"
venv/bin/pip install "triton==2.1.0"

# setup.py leaves these unpinned; current releases (numpy>=2, transformers>=4.32,
# uvloop>=0.18) are incompatible with this torch 2.0.1 / Python 3.9 stack.
log "Pinning numpy/transformers/uvloop to versions compatible with torch 2.0.1"
venv/bin/pip install "numpy==1.26.4" "transformers==4.31.0" "uvloop==0.17.0"

# The S-LoRA security experiments need their own base model at a fixed path:
# huggyllama/llama-7b (public/non-gated), separate from the Llama-3.1-8B used
# elsewhere. Fetched here so they don't fail mid-run_security.sh.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA7B_DIR="$REPO_ROOT/security_eval/malicious_adapter/slora/llama-7b"
if [[ -f "$LLAMA7B_DIR/config.json" ]]; then
    log "huggyllama/llama-7b already downloaded at $LLAMA7B_DIR"
else
    log "Downloading huggyllama/llama-7b (~26GB, public/non-gated) for the S-LoRA security baseline..."
    venv/bin/huggingface-cli download huggyllama/llama-7b --local-dir "$LLAMA7B_DIR"
fi

log "Done"
log "Next: bash scripts/run_security.sh   (from the repo root, runs RQ6)"
