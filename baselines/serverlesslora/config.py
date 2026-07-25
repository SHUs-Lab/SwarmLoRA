# Config for ServerlessLoRA multi-GPU batched inference
import torch

# ============================================================================
# Model Configuration
# ============================================================================

SUPPORTED_MODELS = {
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama-2-13b': 'meta-llama/Llama-2-13b-hf',
}

DEFAULT_LORA_ADAPTERS = {
    'llama-3.1-8b': 'reissbaker/llama-3.1-8b-abliterated-lora',
    'llama-2-7b': 'Styxxxx/llama2_7b_lora-common_gen',
    'mistral-7b': 'Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1488',
    'llama-2-13b': 'Blackroot/Llama-2-13B-Storywriter-LORA',
}

# Active model (change this to switch models)
MODEL_NAME: str = 'llama-3.1-8b'
BASE_MODEL_ID: str = SUPPORTED_MODELS[MODEL_NAME]
LORA_ADAPTER_ID: str = DEFAULT_LORA_ADAPTERS[MODEL_NAME]
USE_LORA: bool = True

# ============================================================================
# Device and Precision
# ============================================================================

WORKER_DEVICE: str = "cuda:0"
DTYPE = torch.bfloat16

# ============================================================================
# Generation Defaults
# ============================================================================

MAX_NEW_TOKENS: int = 1024

# Default ports (used as CLI defaults; overridden by deployment_config.yaml)
BASELINE_SERVER_PORT: int = 50050       # Base model server port
BASELINE_WORKER_BASE_PORT: int = 6000   # Worker base port

# ============================================================================
# ServerlessLoRA Paper Components Configuration
# ============================================================================

# Pre-Loading Scheduler (Section 4.1)
PRELOAD_SCHEDULER_PORT: int = 7100
PRELOAD_SCHEDULE_INTERVAL_MS: int = 1000   # Run scheduling every 1 second
PRELOAD_VALUE_DENSITY_THRESHOLD: float = 0.01  # Minimum density to consider preloading

# Demand-driven rebalancing (Section 4.1 extension)
REBALANCE_RATE_RATIO_THRESHOLD: float = 2.0    # starving rate must be >= 2x overstocked rate
REBALANCE_MIN_CONTAINERS_PER_FUNCTION: int = 1  # never fully evict a function
REBALANCE_COOLDOWN_S: float = 30.0              # min seconds between rebalancing same container
REBALANCE_PRIORITY_OFFSET: int = 1000           # rebalance decisions have lower urgency than preloads
REBALANCE_MAX_SWAPS_PER_NODE: int = 2           # max swap decisions per node per schedule round
REBALANCE_MIN_TOTAL_REQUESTS: int = 10          # min requests in rate window before rebalancing
REBALANCE_MIN_RATE_DIFF: float = 0.05           # min absolute rate diff (req/s) to trigger swap

# Pre-Loading Agent (Section 3.2)
PRELOAD_AGENT_BASE_PORT: int = 7000        # Agent port = base + node_index
CONTAINER_KEEP_ALIVE_MS: int = 60000       # 60 second keep-alive
CONTAINER_SPAWN_TIMEOUT_S: float = 180.0   # Timeout for spawning containers
AGENT_POLL_INTERVAL_MS: int = 500          # Poll scheduler every 500ms

# GPU Offloader (Section 4.3)
GPU_MEMORY_PRESSURE_THRESHOLD: float = 0.90  # Trigger offload at 90% usage
GPU_TARGET_FREE_MB: float = 1024.0         # Target 1GB free after offload
GPU_OFFLOAD_POLL_MS: int = 100             # Check memory every 100ms
GPU_OFFLOAD_COOLDOWN_MS: int = 1000        # Minimum time between offload cycles

# Controller/Frontend (Section 3.2, 3.3)
CONTROLLER_PORT: int = 8000
CONTROLLER_RATE_UPDATE_INTERVAL_S: float = 1.0
CONTROLLER_HEALTH_CHECK_INTERVAL_S: float = 5.0
CONTROLLER_MAX_RETRIES: int = 3

# Offline Profiler (Section 4.2)
PROFILER_BATCH_SIZES: list = [1, 2, 4, 8]
PROFILER_WARMUP_RUNS: int = 3
PROFILER_SAMPLE_RUNS: int = 5
PROFILER_DEFAULT_MAX_TOKENS: int = 32
PROFILER_OUTPUT_FILE: str = "profiles.json"

# Memory estimates (MB) - used for scheduling decisions
MEMORY_ESTIMATES = {
    "container_base_mb": 1000.0,
    "library_torch_mb": 500.0,
    "library_transformers_mb": 100.0,
    "backbone_llama_8b_mb": 16000.0,
    "backbone_llama_7b_mb": 14000.0,
    "backbone_llama_13b_mb": 26000.0,
    "adapter_default_mb": 100.0,
    "kernel_compiled_mb": 20.0,
}

# Loading delay estimates (ms) - used for value computation
LOADING_DELAYS = {
    "library_torch_ms": 2000.0,
    "library_transformers_ms": 500.0,
    "backbone_from_disk_ms": 10000.0,
    "backbone_from_ipc_ms": 100.0,
    "adapter_from_hub_ms": 5000.0,
    "adapter_from_cache_ms": 500.0,
    "adapter_to_gpu_ms": 100.0,
    "kernel_compile_ms": 500.0,
}

# GPU memory default (L40S = 48GB usable ~44400MB; overridden by deployment_config.yaml)
DEFAULT_GPU_MEMORY_MB: float = 44400.0

# Batching defaults (can be overridden by profiler)
DEFAULT_BASE_TTFT_MS: float = 400.0        # Default T_0
DEFAULT_MARGINAL_COST_MS: float = 50.0     # Default α
DEFAULT_SLO_MS: float = 2000.0             # Default SLO target
DEFAULT_MAX_BATCH_SIZE: int = 128           # Safety cap (SLO formula determines actual max)
