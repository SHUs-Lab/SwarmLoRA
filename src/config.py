# Config for multi-GPU split-model LoRA inference
import os
import torch

# Model
BASE_MODEL_ID: str = 'meta-llama/Llama-3.1-8B-Instruct'
LORA_ADAPTER_ID: str = 'reissbaker/llama-3.1-8b-abliterated-lora'

# Devices (defaults — overridden by CLI args in production)
WORKER_DEVICE: str = "cuda:0"
AGGREGATOR_DEVICE: str = "cuda:0"
DTYPE = torch.bfloat16

# GPU Topology
AGGREGATOR_GPU: int = 0                # GPU index for initial aggregator launch
WORKER_GPUS: list = list(range(torch.cuda.device_count()))  # all GPUs visible to this process (restrict via CUDA_VISIBLE_DEVICES)
# Max workers per GPU (8B base model uses ~16 GB on the aggregator).
# Env-overridable so the controller's spawn cap and every aggregator's slot
# pool (aggregator.py NUM_SLOTS) derive from one value. Bounded by
# MAX_SLOTS = 128 (src/cpp_extensions/common/ipc_types.h). 22 matches
# ServerlessLoRA's cap for RQ1 fairness parity.
MAX_WORKERS_PER_GPU: int = int(os.environ.get("MAX_WORKERS_PER_GPU", 22))
if not 1 <= MAX_WORKERS_PER_GPU <= 128:
    raise ValueError(
        f"MAX_WORKERS_PER_GPU={MAX_WORKERS_PER_GPU} outside 1..128 "
        "(MAX_SLOTS compile-time limit in ipc_types.h)"
    )

# IPC buffers
QUEUE_CAPACITY: int = 512
MAX_TICKETS: int = 512
MAX_SEQ_LEN: int = 1024
MAX_WORKERS: int = 1024                # lifetime registration cap per aggregator (IDs are monotonic)

# Ports
DEFAULT_PORT: int = 50056            # aggregator TCP registration
AGGREGATOR_PORTS: dict = {0: 50056, 1: 50057, 2: 50058, 3: 50059}  # per-GPU aggregator ports
WORKER_BASE_PORT: int = 5000         # worker HTTP base
AGGREGATOR_HEALTH_PORT: int = 8000   # aggregator health HTTP
AGGREGATOR_HEALTH_PORTS: dict = {0: 8000, 1: 8001, 2: 8002, 3: 8003}  # per-GPU health ports
CONTROLLER_PORT: int = 8343          # controller HTTP
GLOBAL_CONTROLLER_PORT: int = 8500   # cluster global controller
NODE_AGENT_PORT: int = 9000          # cluster node agent

# One request per worker (inter-adapter scheduling)
MAX_CONCURRENT_REQUESTS_PER_WORKER: int = 1

# Paged Attention
PAGED_ATTENTION_BLOCK_SIZE: int = 16

# Prefill chunking
PREFILL_CHUNK_SIZE: int = int(os.environ.get("PREFILL_CHUNK_SIZE", 16))
