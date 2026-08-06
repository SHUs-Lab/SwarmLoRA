> **Note (SwarmLoRA artifact):** Fork of upstream ServerlessLLM
> (`fu2024serverlessllm`) with added multi-LoRA batching, fractional-GPU
> support, and a trace-replay harness, used for the comparison figures in
> the paper (Figs. 5, 8, 6–7, 9–11, Table II). Trimmed to just that: removed
> upstream's unrelated generic model-loading benchmark suite, `examples/`,
> `tests/`, `configs/`, and most of `docs/` except the README's images.
>
> **Setup:**
> ```bash
> bash baselines/serverlessllm/setup.sh
> ```
> Or manually:
> ```bash
> cd baselines/serverlessllm
> python3 -m venv sllm-env
> sllm-env/bin/pip install --upgrade pip setuptools wheel
> sllm-env/bin/pip install --no-build-isolation -r requirements.txt -r sllm_store/requirements.txt
> sllm-env/bin/pip install "datasets>=3.6.0"
> cd sllm_store && ../sllm-env/bin/python setup.py build_ext --inplace && ../sllm-env/bin/python setup.py develop && cd ..
> cp -P sllm_store/build/lib.linux-x86_64-*/sllm_store/libglog.so* sllm_store/sllm_store/
> sllm-env/bin/python setup.py develop
> ```
> Notes on the above, each found by actually running it (not just reading it):
> - `requirements.txt` lists `serverless-llm-store` (no version pin) from PyPI,
>   but its published sdist is missing the `requirements.txt` its own `setup.py`
>   expects, and building it from build-isolation also chokes on the same
>   isolated-torch-copy issue below. Build the bundled `sllm_store/` directly
>   from source instead (real setup.py, real requirements.txt) -- this is what
>   the commands above do; skip the PyPI package entirely.
> - `pip install --upgrade pip setuptools wheel` first: a bare `python3 -m venv`
>   doesn't bundle `setuptools` on newer Python (3.12+), so `setup.py` fails
>   immediately with `ModuleNotFoundError: No module named 'setuptools'`.
> - `sllm_store/requirements.txt` caps `torch` below 2.11.0: PyPI's plain
>   `torch` wheels switched from bundling CUDA 12.x to CUDA 13.x runtime deps
>   at exactly 2.11.0, which needs a newer driver than many hosts have. The
>   cap keeps the CUDA-12-bundled build for broad reproducibility regardless
>   of the reviewer's driver version.
> - `--no-build-isolation`: pip's default isolated build environment fetches
>   its own fresh torch copy just to run `setup.py`'s build-time checks, and
>   that copy is missing `libcusparseLt.so.0`. Installing torch into the real
>   venv first (already covered by the two `-r` files here) and disabling
>   isolation avoids ever needing that broken isolated copy.
> - `setup.py develop` (not `pip install -e .`): the packaged build backend is
>   missing the `build_editable` hook PEP 660 needs.
> - `sllm_store`'s explicit `build_ext --inplace` before `develop`: on newer
>   setuptools, `setup.py develop` alone silently skips the CMake build
>   entirely (no `build/` directory is even created), so the compiled `.so`
>   extensions never get produced. Running `build_ext --inplace` first forces
>   the actual CMake/ninja build to happen.
> - The `libglog.so*` copy step: `sllm_store`'s own build produces this
>   vendored shared library (visible under `build/lib.../sllm_store/`) but
>   doesn't copy it into the actual package directory, so `sllm-store` (the
>   CLI) and anything importing `sllm_store.server` fails with
>   `OSError: ...libglog.so: cannot open shared object file`. Copy it manually
>   once, right after building.
>
> `sllm/backends/transformers_backend.py` imports `datasets`, which isn't in
> either requirements file above -- installed separately here rather than via
> upstream's `requirements-worker.txt` (`setup.py`'s `extras_require["worker"]`),
> which pins `vllm==0.11.2` -- a version that doesn't exist on PyPI (latest is
> 0.11.0) -- and duplicates several packages (`torch`, `transformers`, `accelerate`,
> `peft`, `ray`) already covered by the other two files. We don't use the vLLM
> backend here (only `--backend transformers`), so there's no reason to pull it in.
> `models/` has Git LFS pointer files, not real weights — `git lfs pull` or
> regenerate adapters before cold-start benchmarks.
>
> **Scripts** (under `scripts/`, run from this directory, named to match the
> main SwarmLoRA-artifact convention; `run_all_traces.sh` is the shared engine
> the other two call into, like `scripts/common.sh` in the main repo):
>
> | Script | Paper element |
> |---|---|
> | `scripts/run_throughput.sh` | Fig. 5 (Poisson throughput sweep) |
> | `scripts/run_cold_start.sh` | Fig. 8 (cold start) — reconstruction, see below |
> | `scripts/run_trace_driven.sh` | Figs. 6–7, 9–11, Table II — 3-min config only (10-min only ever matched exploratory data, dropped) |
>
> No reference results are shipped; each script regenerates output under
> `benchmark_results/`. Trace files are shared with SwarmLoRA and the other
> baselines under the top-level `../../traces/`, not duplicated here.
>
> **`benchmarks/cold_start.py` is a reconstruction, not the original script**
> — the one that produced the paper's cited 1.77s couldn't be recovered
> (exhaustive search of the original research checkout, only its output
> survived). Rebuilds the same measurement via `sllm_store`'s own documented
> interface (`sllm_store/examples/`); should land in the same range but isn't
> guaranteed exact. `test_loading.py`/`benchmark_utils.py` are upstream's own
> related-but-different-scope benchmark (5-replica bulk loading, not
> single-load cold start) — kept for reference.
>
> The rest of this README is ServerlessLLM's own upstream documentation.

<p align="center">
  <picture>
    <img src="./docs/images/serverlessllm.jpg" alt="ServerlessLLM" width="30%">
  </picture>
</p>

<h1 align="center">ServerlessLLM</h1>

<p align="center">
  <strong>Load models 10x faster. Serve 10 models with 1 GPU.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/serverless-llm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/serverless-llm?logo=pypi&logoColor=white&label=PyPI&color=3775A9"></a>
  <a href="https://pypi.org/project/serverless-llm/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/serverless-llm?logo=pypi&logoColor=white&label=Downloads&color=3775A9"></a>
  <a href="https://discord.gg/AEF8Gduvm8"><img alt="Discord" src="https://img.shields.io/discord/1233345500112224279?logo=discord&logoColor=white&label=Discord&color=5865F2"></a>
  <a href="./docs/images/wechat.png"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
</p>

<p align="center">
  <a href="https://serverlessllm.github.io"><b>Docs</b></a> •
  <a href="#-quick-start-90-seconds"><b>Quick Start</b></a> •
  <a href="https://www.usenix.org/conference/osdi24/presentation/fu"><b>OSDI'24 Paper</b></a>
</p>

---

## ⚡ Performance

<!-- <p align="center">
  <img src="./docs/images/benchmark_loading_speed.png" alt="Loading Speed Comparison" width="80%">
</p> -->

**ServerlessLLM loads models 6-10x faster than SafeTensors**, enabling true serverless deployment where multiple models efficiently share GPU resources.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Scenario</th>
      <th>SafeTensors</th>
      <th>ServerlessLLM</th>
      <th>Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen/Qwen3-32B</td>
      <td>Random</td>
      <td>20.6s</td>
      <td>3.2s</td>
      <td><strong>6.40x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>12.5s</td>
      <td>1.3s</td>
      <td><strong>9.95x</strong></td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-R1-Distill-Qwen-32B</td>
      <td>Random</td>
      <td>19.1s</td>
      <td>3.2s</td>
      <td><strong>5.93x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>10.2s</td>
      <td>1.2s</td>
      <td><strong>8.58x</strong></td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>Random</td>
      <td>4.4s</td>
      <td>0.7s</td>
      <td><strong>6.54x</strong></td>
    </tr>
  </tbody>
</table>

*Results obtained on NVIDIA H100 GPUs with NVMe SSD. "Random" simulates serverless multi-model serving; "Cached" shows repeated loading of the same model.*

## What is ServerlessLLM?

ServerlessLLM is a fast, low-cost system for deploying multiple AI models on shared GPUs, with three core innovations:

1. **⚡ Ultra-Fast Checkpoint Loading**: Custom storage format with O_DIRECT I/O loads models 6-10x faster than state-of-the-art checkpoint loaders
2. **🔄 GPU Multiplexing**: Multiple models share GPUs with fast switching and intelligent scheduling
3. **🎯 Unified Inference + Fine-Tuning**: Seamlessly integrates LLM serving with LoRA fine-tuning on shared resources

**Result:** Serve 10 models on 1 GPU, fine-tune on-demand, and serve a base model + 100s of LoRA adapters.

---

## 🚀 Quick Start (90 Seconds)

### Start ServerlessLLM Cluster

> **Don't have Docker?** Jump to [Use the Fast Loader in Your Code](#-use-the-fast-loader-in-your-code) for a Docker-free example.

```bash
# Download the docker-compose.yml file
curl -O https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/examples/docker/docker-compose.yml

# Set model storage location
export MODEL_FOLDER=/path/to/models

# Launch cluster (head node + worker with GPU)
docker compose up -d

# Wait for the cluster to be ready
docker logs -f sllm_head
```

### Deploy a Model

```bash
docker exec sllm_head /opt/conda/envs/head/bin/sllm deploy --model Qwen/Qwen3-0.6B --backend transformers
```

### Query the Model

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is ServerlessLLM?"}],
    "temperature": 0.7
  }'
```

**That's it!** Your model is now serving requests with an OpenAI-compatible API.

---

## 💡 Use the Fast Loader in Your Code

Use ServerlessLLM Store standalone to speed up torch-based model loading.

### Install

```bash
pip install serverless-llm-store
```

### Convert a Model

```bash
sllm-store save --model Qwen/Qwen3-0.6B --backend transformers
```

### Start the Store Server

```bash
# Start the store server first
sllm-store start --storage-path ./models --mem-pool-size 4GB
```

### Load it 6-10x Faster in Your Python Code

```python
from sllm_store.transformers import load_model

# Load model (6-10x faster than from_pretrained!)
model = load_model(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    torch_dtype="float16"
)

# Use as a normal PyTorch/Transformers model
output = model.generate(**inputs)
```

**How it works:**
- Custom binary format optimized for sequential reads
- O_DIRECT I/O bypassing OS page cache
- Pinned memory pool for DMA-accelerated GPU transfers
- Parallel multi-threaded loading

---

## 🎯 Key Features

### ⚡ Ultra-Fast Model Loading
- **6-10x faster** than the SafeTensors checkpoint loader
- Supports both NVIDIA and AMD GPUs
- Works with vLLM, Transformers, and custom models

**📖 Docs:** [Fast Loading Guide](https://serverlessllm.github.io/docs/store/quickstart) | [ROCm Guide](https://serverlessllm.github.io/docs/store/rocm_quickstart)

---

### 🔄 GPU Multiplexing
- **Run 10+ models on 1 GPU** with fast switching
- Storage-aware scheduling minimizes loading time
- Auto-scale instances per model (scale to zero when idle)
- Live migration for zero-downtime resource optimization

**📖 Docs:** [Deployment Guide](https://serverlessllm.github.io/docs/getting_started)

---

### 🎯 Unified Inference + LoRA Fine-Tuning
- Integrates LLM serving with serverless LoRA fine-tuning
- Deploys fine-tuned adapters for inference on-demand
- Serves a base model + 100s of LoRA adapters efficiently

**📖 Docs:** [Fine-Tuning Guide](https://serverlessllm.github.io/docs/features/peft_lora_fine_tuning)

---

### 🔍 Embedding Models for RAG
- Deploy embedding models alongside LLMs
- Provides an OpenAI-compatible `/v1/embeddings` endpoint

**💡 Example:** [RAG Example](https://github.com/ServerlessLLM/ServerlessLLM/tree/main/examples/embedding)

---

### 🚀 Production-Ready
- **OpenAI-compatible API** (drop-in replacement)
- Docker and Kubernetes deployment
- Multi-node clusters with distributed scheduling

**📖 Docs:** [Deployment Guide](https://serverlessllm.github.io/docs/developer/supporting_a_new_hardware) | [API Reference](https://serverlessllm.github.io/docs/api/intro)

---

### 💻 Supported Hardware
- **NVIDIA GPUs**: Compute capability 7.0+ (V100, A100, H100, RTX 3060+)
- **AMD GPUs**: ROCm 6.2+ (MI100, MI200 series) - Experimental

**More Examples:** [ServerlessLLM/examples/](https://github.com/ServerlessLLM/ServerlessLLM/tree/main/examples) (not bundled here -- see the note at the top of this README for what was trimmed)

---

## 🤝 Community

- **Discord**: [Join our community](https://discord.gg/AEF8Gduvm8) - Get help, share ideas
- **GitHub Issues**: [Report bugs](https://github.com/ServerlessLLM/ServerlessLLM/issues)
- **WeChat**: [QR Code](./docs/images/wechat.png) - 中文支持
- **Contributing**: See [CONTRIBUTING.md](https://github.com/ServerlessLLM/ServerlessLLM/blob/main/CONTRIBUTING.md) (upstream; not vendored here)

Maintained by 10+ contributors worldwide. Community contributions are welcome!

---

## 📄 Citation

If you use ServerlessLLM in your research, please cite our [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/fu):

```bibtex
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={OSDI'24},
  year={2024}
}
```

---

## 📝 License

Apache 2.0 - See [LICENSE](./LICENSE)

---

<p align="center">
  <strong>⭐ Star this repo if ServerlessLLM helps you!</strong>
</p>
